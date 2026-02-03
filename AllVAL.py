import torchvision
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import os

imageHR_path = os.listdir('res/hr/')
imageGR_path = os.listdir('res/gr/')


psnr_All_res = []
print('PSNR')
psnr = PSNR()

k = 0

for i in range(len(imageHR_path)):
    imageHR = torchvision.io.read_image("res/hr/"+imageHR_path[i])[0]
    imageHG = torchvision.io.read_image("res/gr/" + imageGR_path[i])[0]
    psnr_All_res.append(psnr(imageHG, imageHR))
    if k == 0:
        print('HG s HR', psnr(imageHG, imageHR))
        print('HR s HR', psnr(imageHR, imageHR))
        k=k+1
# print(psnr_All_res)
print("SUM: ", sum(psnr_All_res))
print("mean PSNR: ", sum(psnr_All_res) / len(imageHR_path))

print('SSIM')
ssim_All_res = []
ssim = SSIM(data_range=255)
from PIL import Image, ImageDraw
for i in range(len(imageHR_path)):
    imageHR = (torchvision.transforms.ToTensor()(Image.open("res/hr/"+imageHR_path[i]))).unsqueeze(0)
    imageHG = torchvision.transforms.ToTensor()(Image.open("res/gr/" + imageGR_path[i])).unsqueeze(0)
    ssim_All_res.append(ssim(imageHG, imageHR))
print(ssim_All_res)
print("SUM: ", sum(ssim_All_res))
print("mean ssim: ", sum(ssim_All_res) / len(imageHR_path))

#FILTERS
import cv2

med_values = []
filters_psnr = {
            'med-3x3': [],
            'med-5x5': [],
            'med-9x9': [],

            'gaus-3x3': [],
            'gaus-5x5': [],
            'gaus-9x9': [],

            'bila-3x3': [],
            'bila-5x5': [],
            'bila-9x9': [],

            }
filters_ssim = {
            'med-3x3': [],
            'med-5x5': [],
            'med-9x9': [],

            'gaus-3x3': [],
            'gaus-5x5': [],
            'gaus-9x9': [],

            'bila-3x3': [],
            'bila-5x5': [],
            'bila-9x9': [],

            }
for i in range(len(imageHR_path)):
    imageHR = torchvision.transforms.ToTensor()(Image.open("res/hr/"+imageHR_path[i])).unsqueeze(0)
    image_gen = cv2.imread("res/gr/" + imageGR_path[i])

    k = 9
    #Медианный фильтр

    med_img_3 = torchvision.transforms.ToTensor()(cv2.medianBlur(image_gen, ksize=3)).unsqueeze(0)
    med_img_5 = torchvision.transforms.ToTensor()(cv2.medianBlur(image_gen, ksize=5)).unsqueeze(0)
    med_img_9 = torchvision.transforms.ToTensor()(cv2.medianBlur(image_gen, ksize=9)).unsqueeze(0)
    filters_psnr['med-3x3'].append(psnr(med_img_3, imageHR))
    filters_psnr['med-5x5'].append(psnr(med_img_5, imageHR))
    filters_psnr['med-9x9'].append(psnr(med_img_9, imageHR))

    filters_ssim['med-3x3'].append(ssim(med_img_3, imageHR))
    filters_ssim['med-5x5'].append(ssim(med_img_5, imageHR))
    filters_ssim['med-9x9'].append(ssim(med_img_9, imageHR))

    #гауссовский фильтр
    gaus_img_3 = torchvision.transforms.ToTensor()(cv2.GaussianBlur(image_gen, (3,3),1)).unsqueeze(0)
    gaus_img_5 = torchvision.transforms.ToTensor()(cv2.GaussianBlur(image_gen, (5,5),1)).unsqueeze(0)
    gaus_img_9 = torchvision.transforms.ToTensor()(cv2.GaussianBlur(image_gen, (9,9),1)).unsqueeze(0)
    filters_psnr['gaus-3x3'].append(psnr(gaus_img_3, imageHR))
    filters_psnr['gaus-5x5'].append(psnr(gaus_img_5, imageHR))
    filters_psnr['gaus-9x9'].append(psnr(gaus_img_9, imageHR))

    filters_ssim['gaus-3x3'].append(ssim(gaus_img_3, imageHR))
    filters_ssim['gaus-5x5'].append(ssim(gaus_img_5, imageHR))
    filters_ssim['gaus-9x9'].append(ssim(gaus_img_9, imageHR))

    #двустороннаяя фильтрация
    bila_img_3 = torchvision.transforms.ToTensor()(cv2.bilateralFilter(image_gen, 3,75,75)).unsqueeze(0)
    bila_img_5 = torchvision.transforms.ToTensor()(cv2.bilateralFilter(image_gen, 5,75,75)).unsqueeze(0)
    bila_img_9 = torchvision.transforms.ToTensor()(cv2.bilateralFilter(image_gen, 9,75,75)).unsqueeze(0)
    filters_psnr['bila-3x3'].append(psnr(bila_img_3, imageHR))
    filters_psnr['bila-5x5'].append(psnr(bila_img_5, imageHR))
    filters_psnr['bila-9x9'].append(psnr(bila_img_9, imageHR))

    filters_ssim['bila-3x3'].append(ssim(bila_img_3, imageHR))
    filters_ssim['bila-5x5'].append(ssim(bila_img_5, imageHR))
    filters_ssim['bila-9x9'].append(ssim(bila_img_9, imageHR))

print('mean psnr med-3x3: ', sum(filters_psnr['med-3x3'])/len(imageHR_path))
print('mean psnr med-5x5: ', sum(filters_psnr['med-5x5'])/len(imageHR_path))
print('mean psnr med-9x9: ', sum(filters_psnr['med-9x9'])/len(imageHR_path))

print('mean psnr gaus-3x3: ', sum(filters_psnr['gaus-3x3'])/len(imageHR_path))
print('mean psnr gaus-5x5: ', sum(filters_psnr['gaus-5x5'])/len(imageHR_path))
print('mean psnr gaus-9x9: ', sum(filters_psnr['gaus-9x9'])/len(imageHR_path))

print('mean psnr bila-3x3: ', sum(filters_psnr['bila-3x3'])/len(imageHR_path))
print('mean psnr bila-5x5: ', sum(filters_psnr['bila-5x5'])/len(imageHR_path))
print('mean psnr bila-9x9: ', sum(filters_psnr['bila-9x9'])/len(imageHR_path))

print('mean ssim med-3x3: ', sum(filters_ssim['med-3x3'])/len(imageHR_path))
print('mean ssim med-5x5: ', sum(filters_ssim['med-5x5'])/len(imageHR_path))
print('mean ssim med-9x9: ', sum(filters_ssim['med-9x9'])/len(imageHR_path))

print('mean ssim gaus-3x3: ', sum(filters_ssim['gaus-3x3'])/len(imageHR_path))
print('mean ssim gaus-5x5: ', sum(filters_ssim['gaus-5x5'])/len(imageHR_path))
print('mean ssim gaus-9x9: ', sum(filters_ssim['gaus-9x9'])/len(imageHR_path))

print('mean ssim bila-3x3: ', sum(filters_ssim['bila-3x3'])/len(imageHR_path))
print('mean ssim bila-5x5: ', sum(filters_ssim['bila-5x5'])/len(imageHR_path))
print('mean ssim bila-9x9: ', sum(filters_ssim['bila-9x9'])/len(imageHR_path))

    #повышение резкости
    # im = cv2.filter2D(image, -1, kernel)