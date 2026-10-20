# GAN-upscale

Увеличение разрешения изображений (super-resolution, x4) с помощью GAN в стиле SRGAN на PyTorch. Для сравнения есть U-Net.

## Файлы
- `main.py` — обучение GAN (генератор с residual-блоками и PixelShuffle, дискриминатор, VGG19 для perceptual loss)
- `U_net.py`, `UnetMod/` — альтернативная модель U-Net
- `AllVAL.py`, `as.py`, `tryPsnr.py` — оценка качества по PSNR / SSIM
- `filters.py` — постобработка (фильтр резкости)
- `pretrained/` — сохранённые веса генератора и дискриминатора

## Запуск
```bash
pip install torch torchvision torchmetrics opencv-python pandas plotly matplotlib scikit-learn tqdm pillow
python main.py
```
Датасет ожидается в `data/`, выход размера 256×256. Параметры (эпохи, lr, batch size) задаются в начале `main.py`.
