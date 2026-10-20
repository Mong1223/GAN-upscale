#
import torch

cuda = torch.cuda.is_available()
print(cuda)

# from tensorflow.python.client import device_lib
# import tensorflow as tf
# # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# print(device_lib.list_local_devices())

# import tensorflow as tf;
# print(tf.config.list_physical_devices('GPU'))



# from PIL import Image
# import cv2
# pil_image = Image.open('images/0_99.png').convert("RGB")
# cv2.imwrite('data/img.png', pil_image)