import numpy as np
import os
import cv2
image_dir = 'res/filter/GaussianBlur_resk'
# f_name = '0_199.png.png'

kernel = np.array([[-1,-1,-1],
          [-1,9,-1],
          [-1,-1,-1]])


for file_name in os.listdir(image_dir):
    image = cv2.imread(os.path.join(image_dir, file_name))
    # image = cv2.imread('res/gr/0_199.png.png')
    # cv2.imshow('rl',image)
    # med_img = image
    k = 9
    #Медианный фильтр
    # med_img = cv2.medianBlur(image, ksize=k)


    #гауссовский фильтр
    gaus_img = cv2.GaussianBlur(image, (k,k),1)
    # cv2.imshow('med',med_img)

    #двустороннаяя фильтрация
    # bila_img = cv2.bilateralFilter (image, k,75,75)


    #повышение резкости
    # im = cv2.filter2D(image, -1, kernel)

    cv2.imwrite('res/filter/GaussianBlur_resk_GaussianBlur/'+file_name+str(k)+'.png', gaus_img)

cv2.waitKey(0)