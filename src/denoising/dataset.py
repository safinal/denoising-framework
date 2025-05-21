import torch
import torchvision
import cv2
import os
import pandas as pd
import numpy as np

from src.config import ConfigManager


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='DataSet2', spreadsheet_file='Labels.xlsx', transform=None):
        self.label_mapping_dict = {'Gaussian': 0, 'Periodic': 1, 'Salt': 2}
        self.annotations = pd.read_excel(os.path.join(root_dir, spreadsheet_file)).drop(columns='Denoise Image')[['Noisy Image', 'Noise Type']]
        self.annotations['Noise Type'] = self.annotations['Noise Type'].map(self.label_mapping_dict)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(filename=img_path, flags=cv2.IMREAD_COLOR)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)


class NoisyImagesDataset(torch.utils.data.Dataset):
    def __init__(self, noise_type, root_dir='DataSet2', spreadsheet_file='Labels.xlsx', transform=None): # 'Gaussian' 'Periodic' 'Salt'
        self.annotations = pd.read_excel(os.path.join(root_dir, spreadsheet_file))
        self.annotations = self.annotations[self.annotations['Noise Type'] == noise_type].drop(columns='Noise Type')[['Noisy Image', 'Denoise Image']]
        self.root_dir = root_dir
        self.transform = transform
        self.noise_type = noise_type

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        noisy_img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        noisy_image = cv2.imread(filename=noisy_img_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        image = cv2.imread(filename=img_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        if self.transform:
            image = self.transform(image)
            noisy_image = self.transform(noisy_image)
        return noisy_image, image


def create_noise_type_detcetion_train_val_datasets_and_loaders():
    dataset = NoiseDataset(transform=torchvision.transforms.ToTensor())
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - ConfigManager().get("val_fraction"), ConfigManager().get("val_fraction")])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=ConfigManager().get("batch_size"), shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=ConfigManager().get("batch_size"), shuffle=True)
    return train_dataset, train_loader, val_dataset, val_loader


def create_denoising_train_val_datasets_and_loaders(noise_type):
    dataset = NoisyImagesDataset(noise_type=noise_type, transform=torchvision.transforms.ToTensor())
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - ConfigManager().get("val_fraction"), ConfigManager().get("val_fraction")])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=ConfigManager().get("batch_size"), shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=ConfigManager().get("batch_size"), shuffle=True)
    return train_dataset, train_loader, val_dataset, val_loader
