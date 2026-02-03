import torchvision
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import os

imageHR_name = os.listdir('res/fullNN/hr')
imageHG_name = os.listdir('res/fullNN/NN')
# print(content)
res = []
print('PSNR')
psnr = PSNR()
for i in range(len(imageHR_name)):
    imageHR = torchvision.io.read_image("res/fullNN/hr/"+imageHR_name[i])[0]
    imageHG = torchvision.io.read_image("res/fullNN/NN/"+imageHG_name[i])[0]
    res.append(psnr(imageHG, imageHR))
print(res)
print("SUM: ", sum(res))
print("mean : ", sum(res)/len(imageHR_name))
print("@@@@@@@@@@@@@@@@@@")
print('SSIM')
res1 = []
ssim = SSIM(data_range=255)
from PIL import Image, ImageDraw
for i in range(len(imageHR_name)):
    imageHR = (torchvision.transforms.ToTensor()(Image.open("res/fullNN/hr/"+imageHR_name[i]))).unsqueeze(0)
    imageHG = torchvision.transforms.ToTensor()(Image.open("res/fullNN/NN/"+imageHG_name[i])).unsqueeze(0)
    res1.append(ssim(imageHG, imageHR))
print(res1)
print("SUM: ", sum(res1))
print("mean : ", sum(res1)/len(imageHR_name))