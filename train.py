import os
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

random.seed(15)
torch.manual_seed(15)

# Веса из pretrained/ обучены со старой (ошибочной) нормализацией и BatchNorm(eps=0.8),
# с исправленной моделью они не совместимы — обучаем заново.
load_pretrained_models = False
pretrained_generator = "saved_models/generator_last.pth"
pretrained_discriminator = "saved_models/discriminator_last.pth"

n_epochs = 10
dataset_path = "data/"
batch_size = 4
lr = 0.0001
b1 = 0.5
b2 = 0.999
n_cpu = 4
hr_height = 256
hr_width = 256
channels = 3
scale = 4
sample_interval = 200  # как часто (в батчах) сохранять пример HR / LR / SR

hr_shape = (hr_height, hr_width)

# Изображения подаются в сеть в диапазоне [-1, 1] — это совпадает с выходом Tanh генератора
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Нормализация, которую ожидает предобученная VGG19
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(x):
    """[-1, 1] -> [0, 1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)


def lr_name_for(hr_path):
    """DIV2K: 0001.png -> 0001x4.png"""
    stem, ext = os.path.splitext(os.path.basename(hr_path))
    return f"{stem}x{scale}{ext}"


def find_pairs(hr_dir, lr_dir):
    """Пары (LR, HR) сопоставляются по имени файла, а не по порядку glob."""
    pairs = []
    for hr_path in sorted(glob.glob(os.path.join(hr_dir, "*.*"))):
        lr_path = os.path.join(lr_dir, lr_name_for(hr_path))
        if os.path.exists(lr_path):
            pairs.append((lr_path, hr_path))
    return pairs


class ImageDataset(Dataset):
    def __init__(self, pairs, hr_shape, train=True):
        self.pairs = pairs
        self.hr_height, self.hr_width = hr_shape
        self.train = train

    def __getitem__(self, index):
        lr_path, hr_path = self.pairs[index]
        img_lr = Image.open(lr_path).convert("RGB")
        img_hr = Image.open(hr_path).convert("RGB")

        # Вырезаем согласованные участки: координаты HR = координаты LR * scale
        lr_h, lr_w = self.hr_height // scale, self.hr_width // scale
        max_x = img_lr.width - lr_w
        max_y = img_lr.height - lr_h
        if self.train:
            x, y = random.randint(0, max_x), random.randint(0, max_y)
        else:
            x, y = max_x // 2, max_y // 2
        img_lr = img_lr.crop((x, y, x + lr_w, y + lr_h))
        img_hr = img_hr.crop((x * scale, y * scale, (x + lr_w) * scale, (y + lr_h) * scale))

        if self.train and random.random() < 0.5:
            img_lr = TF.hflip(img_lr)
            img_hr = TF.hflip(img_hr)

        img_lr = TF.normalize(TF.to_tensor(img_lr), mean, std)
        img_hr = TF.normalize(TF.to_tensor(img_hr), mean, std)
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.pairs)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.register_buffer("mean", imagenet_mean)
        self.register_buffer("std", imagenet_std)

    def forward(self, img):
        # [-1, 1] -> нормализация ImageNet, как при обучении VGG
        img = (denormalize(img) - self.mean) / self.std
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))

        # Upsampling layers (x2 * x2 = x4)
        upsampling = []
        for _ in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer: Tanh -> [-1, 1], тот же диапазон, что и у данных
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


def psnr(sr, hr):
    """PSNR по всем трём каналам, изображения в [-1, 1] переводятся в [0, 255]."""
    sr = denormalize(sr) * 255
    hr = denormalize(hr) * 255
    mse = F.mse_loss(sr, hr, reduction="none").mean(dim=(1, 2, 3))
    return (10 * torch.log10(255 ** 2 / mse.clamp_min(1e-10))).mean().item()


@torch.no_grad()
def validate(generator, dataloader, device):
    generator.eval()
    values = []
    for imgs in dataloader:
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"].to(device)
        values.append(psnr(generator(imgs_lr), imgs_hr))
    generator.train()
    return float(np.mean(values)) if values else float("nan")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    train_pairs = find_pairs(os.path.join(dataset_path, "DIV2K_train_HR"),
                             os.path.join(dataset_path, f"DIV2K_train_LR_bicubic/X{scale}"))
    val_pairs = find_pairs(os.path.join(dataset_path, "DIV2K_valid_HR"),
                           os.path.join(dataset_path, f"DIV2K_valid_LR_bicubic/X{scale}"))
    if not train_pairs:
        raise SystemExit(f"Не найдены пары LR/HR в {dataset_path}")
    print(f"train: {len(train_pairs)} пар, valid: {len(val_pairs)} пар")

    train_dataloader = DataLoader(ImageDataset(train_pairs, hr_shape, train=True),
                                  batch_size=batch_size, shuffle=True, num_workers=n_cpu, drop_last=True)
    val_dataloader = DataLoader(ImageDataset(val_pairs, hr_shape, train=False),
                                batch_size=1, shuffle=False, num_workers=n_cpu)

    # Initialize generator and discriminator
    generator = GeneratorResNet().to(device)
    discriminator = Discriminator(input_shape=(channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device).eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if load_pretrained_models:
        generator.load_state_dict(torch.load(pretrained_generator, map_location=device))
        discriminator.load_state_dict(torch.load(pretrained_discriminator, map_location=device))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    best_psnr = -float("inf")

    for epoch in range(n_epochs):
        gen_loss, disc_loss = 0, 0
        tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=len(train_dataloader))
        for batch_idx, imgs in enumerate(tqdm_bar):
            imgs_lr = imgs["lr"].to(device)
            imgs_hr = imgs["hr"].to(device)

            # Adversarial ground truths
            valid = torch.ones((imgs_lr.size(0), *discriminator.output_shape), device=device)
            fake = torch.zeros((imgs_lr.size(0), *discriminator.output_shape), device=device)

            ### Train Generator
            optimizer_G.zero_grad(set_to_none=True)

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features)

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            optimizer_G.step()

            ### Train Discriminator
            optimizer_D.zero_grad(set_to_none=True)
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            gen_loss += loss_G.item()
            disc_loss += loss_D.item()
            tqdm_bar.set_postfix(gen_loss=gen_loss / (batch_idx + 1), disc_loss=disc_loss / (batch_idx + 1))

            # Пример HR / LR (bicubic x4) / SR — в реальных цветах, без min-max нормализации
            if batch_idx % sample_interval == 0:
                with torch.no_grad():
                    lr_up = F.interpolate(imgs_lr[:1], scale_factor=scale, mode="bicubic", align_corners=False)
                    img_grid = torch.cat((denormalize(imgs_hr[:1]), denormalize(lr_up), denormalize(gen_hr[:1])), -1)
                save_image(img_grid, f"images/{epoch}_{batch_idx}.png")

        # Чекпоинты: последняя модель и лучшая по PSNR на валидации
        torch.save(generator.state_dict(), "saved_models/generator_last.pth")
        torch.save(discriminator.state_dict(), "saved_models/discriminator_last.pth")
        val_psnr = validate(generator, val_dataloader, device)
        print(f"Epoch {epoch}: val PSNR = {val_psnr:.2f} dB")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(generator.state_dict(), "saved_models/generator_best.pth")


if __name__ == '__main__':
    main()
