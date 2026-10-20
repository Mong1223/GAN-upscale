"""Режет примеры из images/ (HR | LR bicubic | SR, см. train.py) на три папки.

Ширина каждой панели равна высоте картинки, так что нарезка не зависит от hr_height.
"""
import cv2
import os

image_dir = "images"
out_dirs = ["res/hr", "res/lr", "res/gr"]   # порядок панелей в images/
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

for file_name in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, file_name))
    size = img.shape[0]
    if img.shape[1] != 3 * size:
        print(f"{file_name}: неожиданный размер {img.shape[:2]}, пропущено")
        continue
    for i, out_dir in enumerate(out_dirs):
        cv2.imwrite(os.path.join(out_dir, file_name), img[:, i * size:(i + 1) * size])
