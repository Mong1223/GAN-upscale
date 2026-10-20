"""PSNR / SSIM между эталонами (HR) и результатами модели.

Пары сопоставляются по имени файла (без расширения), метрики считаются
по всем трём каналам RGB на реальных пикселях.

python evaluate.py --hr data/DIV2K_valid_HR --sr res/gan
python evaluate.py --hr data/DIV2K_valid_HR --sr res/gan --filters
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

# Постобработка результата классическими фильтрами (на RGB-массиве uint8)
FILTERS = {
    "med-3x3": lambda im: cv2.medianBlur(im, 3),
    "med-5x5": lambda im: cv2.medianBlur(im, 5),
    "med-9x9": lambda im: cv2.medianBlur(im, 9),
    "gaus-3x3": lambda im: cv2.GaussianBlur(im, (3, 3), 1),
    "gaus-5x5": lambda im: cv2.GaussianBlur(im, (5, 5), 1),
    "gaus-9x9": lambda im: cv2.GaussianBlur(im, (9, 9), 1),
    "bila-3x3": lambda im: cv2.bilateralFilter(im, 3, 75, 75),
    "bila-5x5": lambda im: cv2.bilateralFilter(im, 5, 75, 75),
    "bila-9x9": lambda im: cv2.bilateralFilter(im, 9, 75, 75),
    "sharpen": lambda im: cv2.filter2D(im, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])),
}


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img):
    """uint8 HxWx3 -> float 1x3xHxW в [0, 1]"""
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255


def metrics(sr, hr):
    sr_t, hr_t = to_tensor(sr), to_tensor(hr)
    psnr = peak_signal_noise_ratio(sr_t, hr_t, data_range=1.0).item()
    ssim = structural_similarity_index_measure(sr_t, hr_t, data_range=1.0).item()
    return psnr, ssim


def find_pairs(hr_dir, sr_dir):
    hr_files = {os.path.splitext(n)[0]: n for n in os.listdir(hr_dir)}
    pairs = []
    for name in sorted(os.listdir(sr_dir)):
        stem = os.path.splitext(name)[0]
        if stem in hr_files:
            pairs.append((os.path.join(hr_dir, hr_files[stem]), os.path.join(sr_dir, name)))
        else:
            print(f"нет эталона для {name}, пропущено")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr", required=True, help="папка с эталонными изображениями")
    parser.add_argument("--sr", required=True, help="папка с результатами модели")
    parser.add_argument("--filters", action="store_true", help="также оценить результат после фильтров")
    args = parser.parse_args()

    pairs = find_pairs(args.hr, args.sr)
    if not pairs:
        raise SystemExit("Не найдено ни одной пары с одинаковыми именами")

    results = {"model": []}
    if args.filters:
        results.update({name: [] for name in FILTERS})

    for hr_path, sr_path in pairs:
        hr, sr = load_rgb(hr_path), load_rgb(sr_path)
        if hr.shape != sr.shape:
            raise SystemExit(f"Размеры не совпадают: {hr_path} {hr.shape} vs {sr_path} {sr.shape}")
        results["model"].append(metrics(sr, hr))
        if args.filters:
            for name, f in FILTERS.items():
                results[name].append(metrics(f(sr), hr))

    print(f"Пар: {len(pairs)}")
    print(f"{'вариант':<10} {'PSNR, dB':>9} {'SSIM':>7}")
    for name, values in results.items():
        psnr, ssim = np.mean(values, axis=0)
        print(f"{name:<10} {psnr:>9.2f} {ssim:>7.4f}")


if __name__ == "__main__":
    main()
