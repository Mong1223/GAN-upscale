"""Применяет фильтр OpenCV ко всем изображениям папки (постобработка результата GAN).

python filter_images.py --input res/gan --output res/gan_gaus-9x9 --filter gaus-9x9
"""
import argparse
import os

import cv2

from evaluate import FILTERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filter", required=True, choices=sorted(FILTERS))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    f = FILTERS[args.filter]
    for name in sorted(os.listdir(args.input)):
        image = cv2.imread(os.path.join(args.input, name))
        if image is None:
            continue
        cv2.imwrite(os.path.join(args.output, name), f(image))


if __name__ == "__main__":
    main()
