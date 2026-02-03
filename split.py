import cv2
import os


image_dir = "images"
train_data = []
for file_name in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, file_name))
    y = 0
    x = 0
    h = 512
    w = 512
    hr_img = img[x:w, y:h]
    cv2.imwrite(f'res/hr/{file_name}.png', hr_img)
    y = 512
    x = 0
    h = 1024
    w = 512
    lr_img = img[x:w, y:h]
    cv2.imwrite(f'res/lr/{file_name}.png', lr_img)
    y = 1024
    x = 0
    h = 1536
    w = 512
    gr_img = img[x:w, y:h]
    cv2.imwrite(f'res/gr/{file_name}.png', gr_img)
# image = cv2.imread(r"images/0_99.png")
# y=512
# x=0
# h=1024
# w=512
# crop_image = image[x:w, y:h]
# cv2.imshow("Cropped", crop_image)
cv2.waitKey(0)