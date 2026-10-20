# GAN-upscale

Увеличение разрешения изображений (super-resolution, x4) с помощью GAN в стиле SRGAN на PyTorch + U-Net для доработки результата.

## Файлы
- `train.py` — обучение GAN (генератор с residual-блоками и PixelShuffle, дискриминатор, VGG19 для perceptual loss), валидация по PSNR после каждой эпохи
- `infer.py` — апскейл папки изображений обученным генератором
- `evaluate.py` — PSNR / SSIM результата относительно эталона (опционально — после фильтров OpenCV)
- `split_samples.py` — нарезка примеров из `images/` на `res/hr`, `res/lr`, `res/gr`
- `unet.py` — U-Net (Keras), дорабатывающая выход GAN; train/test разделены
- `sharpen.py` — фильтр резкости для папки изображений
- `pixel_mlp_rgb.py`, `pixel_mlp_channels.py` — ранний эксперимент на Keras: полносвязная сеть по окрестности 3×3 пикселя (RGB целиком / по каналам)
- `check_gpu.py` — проверка доступности CUDA
- `saved_models/` — `unet.h5`, `pixel_mlp.h5`, `pixel_mlp_v1.h5` (старые веса Keras-моделей); сюда же `train.py` сохраняет новые веса
- `pretrained/` — веса старой версии GAN (несовместимы с исправленной, см. ниже)

## Данные
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), структура:
```
data/DIV2K_train_HR/            0001.png ...
data/DIV2K_train_LR_bicubic/X4/ 0001x4.png ...
data/DIV2K_valid_HR/
data/DIV2K_valid_LR_bicubic/X4/
```

## Запуск
```bash
pip install -r requirements.txt
python train.py                                   # обучение -> saved_models/
python infer.py --weights saved_models/generator_best.pth --input data/DIV2K_valid_LR_bicubic/X4 --output res/gan
python evaluate.py --hr data/DIV2K_valid_HR --sr res/gan --filters
```
Параметры (эпохи, lr, batch size, размер кропа) задаются в начале `train.py`.

## Исправленные ошибки
- Данные нормализовались по ImageNet (≈ [-2.1, 2.6]), а генератор с `Tanh` выдаёт [-1, 1] — теперь данные в [-1, 1], а для VGG нормализация ImageNet делается внутри `FeatureExtractor`.
- `BatchNorm2d(n, 0.8)` задавал `eps=0.8` вместо `momentum` — убрано.
- LR/HR пары сопоставлялись по порядку `glob`, а кропы могли не совпадать — теперь пары по имени, кропы по согласованным координатам.
- Чекпоинты перезаписывали `pretrained/` и выбирались по loss одного батча — теперь `saved_models/generator_last.pth` и `generator_best.pth` (по PSNR на валидации).
- Метрики считались на обучающих картинках, после min-max нормализации, только по красному каналу, SSIM с неверным `data_range`, пары подбирались по порядку `listdir` — всё исправлено в `evaluate.py`.
- U-Net обучалась и проверялась на одних и тех же картинках — теперь отдельная тестовая часть.

Веса в `pretrained/` и `saved_models/unet.h5` получены со старым кодом, поэтому модели нужно переобучить.
