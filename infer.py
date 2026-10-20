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

from train import GeneratorResNet, denormalize, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True, help="папка с LR изображениями")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorResNet().to(device)
    generator.load_state_dict(torch.load(args.weights, map_location=device))
    generator.eval()

    os.makedirs(args.output, exist_ok=True)
    for name in sorted(os.listdir(args.input)):
        img = Image.open(os.path.join(args.input, name)).convert("RGB")
        x = TF.normalize(TF.to_tensor(img), mean, std).unsqueeze(0).to(device)
        with torch.no_grad():
            sr = generator(x)
        stem, _ = os.path.splitext(name)
        out_name = re.sub(r"x\d+$", "", stem) + ".png"
        save_image(denormalize(sr), os.path.join(args.output, out_name))
        print(out_name)


if __name__ == "__main__":
    main()
