import keras.models
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.optimizers import Adam



#Определяем функцию заргузки изображений:
def download_data(path):
  data = []
  for path_image in sorted(os.listdir(path=path)):
    image = Image.open(path + path_image) #Открываем изображение.
    data.append(np.array(image)) #Загружаем пиксели.
  return data


##Model
inputs = Input([512, 512, 3])
#свертка
conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
#максимальное объединение
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#свертка
conv2 = Conv2D(64, 5, activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, 5, activation='relu', padding='same')(conv2)
#максимальное объединение
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#свертка
conv3 = Conv2D(128, 7, activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, 7, activation='relu', padding='same')(conv3)

#расширение
up1 = UpSampling2D(size=(2,2))(conv3)
#конкатенация
merge1 = concatenate([conv2, up1], axis=3)
#свертка
conv4 = Conv2D(64, 5, activation='relu', padding='same')(merge1)
conv4 = Conv2D(64, 5, activation='relu', padding='same')(conv4)

#расширение
up2 = UpSampling2D(size=(2,2))(conv4)
#конкатенация
merge1 = concatenate([conv1, up2], axis = 3)
#свертка
conv5 = Conv2D(32, 3, activation='relu', padding='same')(merge1)
conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv5)

#выходной слой
conv6 = Conv2D(3, 3, activation=None, padding='same')(conv5)





if __name__ == '__main__':
    ##load Data
    X_train = download_data("res/gr/")
    Y_train = download_data("res/hr/")
    X_test = download_data("res/gr/")
    Y_test = download_data("res/hr/")

    # обработка данных [0,1]
    X_train_pred = np.array(X_train).reshape([88, 512, 512, 3]) / 255
    Y_train_pred = np.array(Y_train).reshape([88, 512, 512, 3]) / 255

    X_test_pred = np.array(X_test).reshape([88, 512, 512, 3]) / 255
    Y_test_pred = np.array(Y_test).reshape([88, 512, 512, 3]) / 255

    #train 3033 loss: 0.0016

    ## Model
    model = Model(inputs=inputs, outputs=conv6)
    model.compile(optimizer=Adam(), loss='MeanSquaredError')

    load_use = True
    if load_use:
        model = keras.models.load_model('UnetMod/nn.h5')
    ## Train
    model.fit(X_train_pred, Y_train_pred, batch_size=1, epochs=500)
    model.evaluate(X_test_pred, Y_test_pred, batch_size=1)
    model.save('UnetMod/nn.h5')
    out = model.predict(X_test_pred, batch_size=1)
    # I = 1 # номер картинки после обработки нейронной сетью
    for l in range(88):
        plt.imshow(np.array(out[l]))
        save_fn = 'res/UNET2/{:d}'.format(l) + '.png'
        plt.savefig(save_fn, dpi=160, bbox_inches='tight')
        im_rgb = cv2.cvtColor(np.array(out[l])*255, cv2.COLOR_BGR2RGB)

        cv2.imwrite('res/UNET_cv2/{:d}'.format(l) + '.png', im_rgb)