"""U-Net для доработки результата GAN: вход — res/gr (выход GAN), цель — res/hr.

Выборка делится на train / test, метрики считаются только на test.
Результаты сохраняются под теми же именами, что и входы, для evaluate.py:

python evaluate.py --hr res/hr_test --sr res/UNET
"""
import os
import random
import shutil

import numpy as np
from PIL import Image
import keras.models
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam

gr_dir = "res/gr/"
hr_dir = "res/hr/"
out_dir = "res/UNET/"
hr_test_dir = "res/hr_test/"   # эталоны только тестовой части — для evaluate.py
model_path = "saved_models/unet.h5"
test_fraction = 0.2
epochs = 500
load_use = False   # старая saved_models/unet.h5 обучалась на тестовых картинках — не использовать


def load_images(path, names):
    """Пиксели в [0, 1]"""
    return np.stack([np.array(Image.open(os.path.join(path, n)).convert("RGB")) for n in names]) / 255


def build_model(height, width):
    inputs = Input([height, width, 3])
    # свертка
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    # максимальное объединение
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 5, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 5, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 7, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 7, activation='relu', padding='same')(conv3)

    # расширение + конкатенация
    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(64, 5, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(64, 5, activation='relu', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv5)

    # выходной слой
    conv6 = Conv2D(3, 3, activation=None, padding='same')(conv5)

    model = Model(inputs=inputs, outputs=conv6)
    model.compile(optimizer=Adam(), loss='MeanSquaredError')
    return model


if __name__ == '__main__':
    # Берём только файлы, у которых есть пара с тем же именем
    names = sorted(set(os.listdir(gr_dir)) & set(os.listdir(hr_dir)))
    random.Random(15).shuffle(names)
    n_test = max(1, int(len(names) * test_fraction))
    test_names, train_names = sorted(names[:n_test]), sorted(names[n_test:])
    print(f"train: {len(train_names)}, test: {len(test_names)}")

    X_train, Y_train = load_images(gr_dir, train_names), load_images(hr_dir, train_names)
    X_test, Y_test = load_images(gr_dir, test_names), load_images(hr_dir, test_names)

    if load_use:
        model = keras.models.load_model(model_path)
    else:
        model = build_model(*X_train.shape[1:3])

    model.fit(X_train, Y_train, batch_size=1, epochs=epochs, validation_data=(X_test, Y_test))
    print("test MSE:", model.evaluate(X_test, Y_test, batch_size=1))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(hr_test_dir, exist_ok=True)
    out = model.predict(X_test, batch_size=1)
    for name, img in zip(test_names, out):
        img = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(img).save(os.path.join(out_dir, name))
        shutil.copy(os.path.join(hr_dir, name), os.path.join(hr_test_dir, name))
