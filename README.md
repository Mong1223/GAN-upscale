# GAN-upscale

Увеличение разрешения изображений (super-resolution, x4) с помощью GAN в стиле SRGAN на PyTorch + U-Net для доработки результата.

## Файлы
- `train.py` — обучение: сначала предобучение генератора на попиксельной ошибке, затем GAN (генератор с residual-блоками и PixelShuffle, дискриминатор, VGG19 для perceptual loss); валидация по PSNR после каждой эпохи
- `infer.py` — апскейл папки изображений обученным генератором
- `evaluate.py` — PSNR / SSIM результата относительно эталона (опционально — после фильтров OpenCV)
- `split_samples.py` — нарезка примеров из `images/` на `res/hr`, `res/lr`, `res/gr`
- `unet.py` — U-Net (Keras), дорабатывающая выход GAN; train/test разделены
- `filter_images.py` — применить фильтр OpenCV (медианный, гауссов, двусторонний, резкость) ко всем изображениям папки
- `pixel_mlp_rgb.py`, `pixel_mlp_channels.py` — ранний эксперимент на Keras: полносвязная сеть по окрестности 3×3 пикселя (RGB целиком / по каналам)
- `check_gpu.py` — проверка, видит ли PyTorch видеокарту
- `saved_models/` — `unet.h5`, `pixel_mlp.h5`, `pixel_mlp_v1.h5` (старые веса Keras-моделей); сюда же `train.py` сохраняет новые веса
- `pretrained/` — веса старой версии GAN (несовместимы с исправленной, см. ниже)

## Данные
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) в папке `data/` внутри проекта (пути в `train.py`, `unet.py`, `split_samples.py` считаются от папки проекта, поэтому скрипты можно запускать из любой директории):
```
data/DIV2K_train_HR/            0001.png ...
data/DIV2K_train_LR_bicubic/X4/ 0001x4.png ...
data/DIV2K_valid_HR/
data/DIV2K_valid_LR_bicubic/X4/
```

## Видеокарта
Обучать на CPU не имеет смысла — нужна NVIDIA GPU и PyTorch, собранный с CUDA. Обычный `pip install torch` на Windows ставит CPU-версию, поэтому сначала:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Затем проверка:
```bash
python check_gpu.py
```
Ожидаемый вывод — `Сборка с CUDA: 12.6`, имя видеокарты и `Тест вычислений на GPU: OK`. Если вместо этого `Сборка с CUDA: нет` — установлена CPU-версия, переустановите командой выше. Если сборка с CUDA есть, но GPU не найдена — проверьте драйвер командой `nvidia-smi`.

`train.py` при старте тоже печатает, на чём идёт обучение (`GPU: ...`), и предупреждает, если видеокарта недоступна.

## Запуск
```bash
pip install -r requirements.txt
python train.py                                   # обучение -> saved_models/
python infer.py --weights saved_models/generator_best.pth --input data/DIV2K_valid_LR_bicubic/X4 --output res/gan
python evaluate.py --hr data/DIV2K_valid_HR --sr res/gan --filters
```
Параметры задаются в начале `train.py`: `pretrain_epochs` (20) — предобучение, `n_epochs` (30) — GAN, а также lr, batch size, размер кропа. На RTX 3060 Laptop эпоха занимает ~1 минуту при ~1.5 ГБ видеопамяти, всё обучение — около часа.

Веса сохраняются в `saved_models/`:
- `generator_pretrain.pth` — после предобучения (максимальный PSNR, но картинка мягче);
- `generator_best.pth` — лучший по PSNR в GAN-фазе;
- `generator_last.pth`, `discriminator_last.pth` — последняя эпоха (для продолжения обучения: `load_pretrained_models = True`). При первом запуске скачиваются веса VGG19 (~550 МБ).

`infer.py` обрабатывает большие картинки кусками (`--tile 128` пикселей LR) — так полноразмерные изображения DIV2K помещаются в память GPU. Если памяти всё равно не хватает, уменьшите `--tile`.

## Исправленные ошибки
- Генератор учился только по признакам VGG — цвета ничем не ограничивались, на выходе была зелёная рябь. Добавлены предобучение на попиксельной ошибке (как в статье SRGAN) и попиксельная составляющая в GAN-фазе.
- С BatchNorm и выходом «картинка целиком» генератор учился хуже бикубической интерполяции (21 дБ против 25). Теперь генератор без BatchNorm (как в EDSR) и предсказывает только детали поверх бикубического увеличения — за 3 эпохи предобучения ~25.9 дБ на валидации.
- Данные нормализовались по ImageNet (≈ [-2.1, 2.6]), а генератор с `Tanh` выдаёт [-1, 1] — теперь данные в [-1, 1], а для VGG нормализация ImageNet делается внутри `FeatureExtractor`.
- `BatchNorm2d(n, 0.8)` задавал `eps=0.8` вместо `momentum` — в дискриминаторе исправлено, из генератора BatchNorm убран.
- LR/HR пары сопоставлялись по порядку `glob`, а кропы могли не совпадать — теперь пары по имени, кропы по согласованным координатам.
- Чекпоинты перезаписывали `pretrained/` и выбирались по loss одного батча — теперь `saved_models/generator_last.pth` и `generator_best.pth` (по PSNR на валидации).
- Метрики считались на обучающих картинках, после min-max нормализации, только по красному каналу, SSIM с неверным `data_range`, пары подбирались по порядку `listdir` — всё исправлено в `evaluate.py`.
- U-Net обучалась и проверялась на одних и тех же картинках — теперь отдельная тестовая часть.

Веса в `pretrained/` и `saved_models/unet.h5` получены со старым кодом, поэтому модели нужно переобучить.
