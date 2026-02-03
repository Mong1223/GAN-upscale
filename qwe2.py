import numpy as np
import torch
import torchvision
import os
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os

import keras.backend
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras import metrics
from keras.layers import LeakyReLU, ReLU
import cv2
import matplotlib.pyplot as plt
imageHR_path = os.listdir('res/hr/')
imageGR_path = os.listdir('res/gr/')
#полносвязная
import cv2

i = cv2.imread("res/nntest/" + imageHR_path[0])
i_t = torch.tensor(i)
print(i_t.shape)
class nnr:
    def __init__(self, train_data_path, train_data_pathy):
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
        sess = tf.compat.v1.Session(config=config)
        keras.backend.set_session(sess)
        self.make_data(train_data_path, train_data_pathy)
        self.buldFeedForward()
        self.compileModel()
        load = True
        if load == True:
            self.model.load_weights('saved_models/myModel.h5')
    def make_data(self,path_test,path_lab):
        train_data=[]
        train_data_labels=[]
        for name in os.listdir(path_test):
            img = cv2.imread(os.path.join(path_test,name))
            img= cv2.copyMakeBorder(img,1,1,1,1,  cv2.BORDER_CONSTANT, None,value=0)

            b, g, r = cv2.split(img)
            # image_b = nnr.convMake(b, 3, 3)
            # image_g = nnr.convMake(g, 3, 3)
            # image_r = nnr.convMake(r, 3, 3)
            train_data.append(nnr.convMake(b, 3, 3))
            train_data.append(nnr.convMake(g, 3, 3))
            train_data.append(nnr.convMake(r, 3, 3))
            # train_data.append(nnr.convMake(img,3, 3))

        for file_name in os.listdir(path_lab):
            img = cv2.imread(os.path.join(path_lab, file_name))
            b, g, r = cv2.split(img)
            train_data_labels.append(nnr.my_split(b, 1, 1))
            train_data_labels.append(nnr.my_split(g, 1, 1))
            train_data_labels.append(nnr.my_split(r, 1, 1))
            # train_data_labels.append(nnr.my_split(img, 1, 1))


        # self.train_data = train_data
        self.train_data = np.array(train_data)
        self.train_data = self.train_data.reshape((self.train_data.shape[0] * self.train_data.shape[1],
                                                   self.train_data.shape[2], self.train_data.shape[3]))
        # self.train_data_labels = train_data_labels

        self.train_data_labels = np.array(train_data_labels)
        self.train_data_labels = self.train_data_labels.reshape((self.train_data_labels.shape[0] *
                                                                 self.train_data_labels.shape[1],
                                                                 self.train_data_labels.shape[2],
                                                                 self.train_data_labels.shape[3]))

    @staticmethod
    def convMake(img,rows,cols):
        ret = np.zeros(((img.shape[0] - rows + 1) * (img.shape[1] - cols + 1), 3, 3))
        for i in range(0, img.shape[0] - rows + 1):
            for j in range(0, img.shape[1] - cols + 1):
                ret[i * (img.shape[1] - rows + 1) + j] = img[i:i + rows, j:j + cols]
        return ret

    @staticmethod
    def my_split(array, rows, cols):
        r, h = array.shape
        return (array.reshape(h // rows, rows, -1, cols,)
                    .swapaxes(1, 2)
                    .reshape(-1, rows, cols))


    # self.train_d, train_l = make_data('res/nntest/', 'res/nnlab/')
    # print('asdas')
    # print(i.size())

    # test = i[0:3, 0:3]
    # print(test.size)
    # test2 = test.reshape((27,))
    # print(test2.size)

    # N= 2 #batch size
    # D_in= [3,3,3]#вход
    # H=
    # D_Out = 64,2,100,3
    #
    # two_layer_net = torch.nn.Sequential(
    #     torch.nn.Linear(D_in,H),
    #     torch.nn.PReLU(),
    #     torch.nn.Linear(H,D_Out),
    #     torch.nn.PReLU()
    # )



    def buldFeedForward(self):
        input_layer = Input(shape=(3, 3))

        x = Flatten()(input_layer)
        x = Dense(units=9)(x)
        # x = Activation('linear')(x)
        x = ReLU()(x)
        x = Dense(units=9)(x)
        # x = Activation('linear')(x)
        x = ReLU()(x)
        x = Dense(units=3)(x)
        # x = Activation('linear')(x)
        x = ReLU()(x)
        x = Dense(units=1)(x)
        # output_layer = Activation('linear')(x)
        output_layer = ReLU()(x)
        # LeakyReLU
        self.model = Model(input_layer, output_layer)


    def compileModel(self):
        opt = Adam(lr=0.05)
        self.model.compile(
            optimizer=opt,
            loss='mean_squared_logarithmic_error',
            metrics=['accuracy']
        )


    def trainModel(self):
        self.model.fit(self.train_data,
                       self.train_data_labels,
                       batch_size=1000,
                       epochs=4,
                       shuffle=True)


    def predict(self, image_path):
        img = cv2.imread(image_path)
        image_shape = img.shape
        image_shape = [image_shape[0],image_shape[1]]
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=0)
        b,g,r = cv2.split(img)

        # image_b = nnr.convMake(b, 3, 3)
        # image_g = nnr.convMake(g, 3, 3)
        # image_r = nnr.convMake(r, 3, 3)
        image_b = np.array(nnr.convMake(b, 3, 3))
        image_g = np.array(nnr.convMake(g, 3, 3))
        image_r = np.array(nnr.convMake(r, 3, 3))


        # image = nnr.convMake(img, 3, 3)
        # image = np.array(image)
        prediction_b = self.model.predict(image_b)
        prediction_g = self.model.predict(image_g)
        prediction_r = self.model.predict(image_r)
        www = self.model.weights
        self.model.save('saved_models/myModel.h5')
        pred_image_b = prediction_b.reshape(image_shape)
        pred_image_g = prediction_g.reshape(image_shape)
        pred_image_r = prediction_r.reshape(image_shape)
        pred_image = cv2.merge([pred_image_b,pred_image_g,pred_image_r])
        print('www',www)
        return pred_image

if __name__ == "__main__":
    image_dirx = "res/nntest"
    image_diry = "res/nnlab"
    improver = nnr(image_dirx, image_diry)
    improver.trainModel()


    # absolutely_random_image = os.path.join(image_dir, os.listdir(image_dir)[11])
    # sh, bl, image,image2 = improver.predict('testim/0013x4.png')
    # sh = Image.fromarray(sh)
    # bl = Image.fromarray(bl)
    # image = Image.fromarray(image)
    # image2 = Image.fromarray(image2)
    # sh.show()
    # bl.show()
    # image.show()
    # image2.show()


    aa = improver.predict('res/gr/3_299.png.png')
    file_name = '3_299.png'
    cv2.imwrite(f'res/NN/{file_name}.png', aa)
    # image = Image.fromarray(aa).convert('RGB')
    # image.save('res/NN/'+str(aa)+'.jpeg')
