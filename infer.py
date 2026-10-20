"""Апскейл изображений обученным генератором.

Результат сохраняется под именем HR-оригинала (0801x4.png -> 0801.png),
чтобы evaluate.py сопоставлял пары по имени.

python infer.py --weights saved_models/generator_best.pth \
    --input data/DIV2K_valid_LR_bicubic/X4 --output res/gan
"""
import argparse
import os
import re

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

from train import GeneratorResNet, denormalize, mean, std, scale


@torch.no_grad()
def upscale(generator, x, tile, overlap=16):
    """Апскейл по кускам tile x tile (в пикселях LR), чтобы большие картинки помещались в память GPU.

    Каждый кусок берётся с запасом overlap по краям, в результат идёт только центральная часть —
    так на стыках нет швов от краевых эффектов свёрток.
    """
    _, _, h, w = x.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return generator(x)
    out = torch.zeros((x.shape[0], 3, h * scale, w * scale), device=x.device)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - overlap, 0), max(x0 - overlap, 0)
            py1, px1 = min(y1 + overlap, h), min(x1 + overlap, w)
            sr = generator(x[:, :, py0:py1, px0:px1])
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = sr[
                :, :, (y0 - py0) * scale:(y1 - py0) * scale, (x0 - px0) * scale:(x1 - px0) * scale]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True, help="папка с LR изображениями")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile", type=int, default=128,
                        help="размер куска в пикселях LR (0 — целиком; уменьшите, если не хватает памяти GPU)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorResNet().to(device)
    generator.load_state_dict(torch.load(args.weights, map_location=device))
    generator.eval()

    os.makedirs(args.output, exist_ok=True)
    for name in sorted(os.listdir(args.input)):
        img = Image.open(os.path.join(args.input, name)).convert("RGB")
        x = TF.normalize(TF.to_tensor(img), mean, std).unsqueeze(0).to(device)
        sr = upscale(generator, x, args.tile)
        stem, _ = os.path.splitext(name)
        out_name = re.sub(r"x\d+$", "", stem) + ".png"
        save_image(denormalize(sr), os.path.join(args.output, out_name))
        print(out_name)


if __name__ == "__main__":
    main()
