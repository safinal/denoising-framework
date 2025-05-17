import torch
import torchvision
import os
import cv2
import pandas as pd

from src.defect_detection.config import batch_size


class DisksDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        defected_images_names = list(map(lambda x: os.path.join('Defected', x), os.listdir(os.path.join(root_dir, 'Defected'))))
        ok_images_names = list(map(lambda x: os.path.join('OK', x), os.listdir(os.path.join(root_dir, 'OK'))))
        self.annotations = pd.DataFrame({'img_name': defected_images_names + ok_images_names, 'label': [1]*len(defected_images_names) + [0]*len(ok_images_names)})
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


def create_defect_detection_train_val_datasets_and_loaders(dataset_root_dir, val_fraction=0.2):
    dataset = DisksDataset(root_dir=dataset_root_dir, transform=torchvision.transforms.ToTensor())
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - val_fraction, val_fraction])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    return dataset, train_dataset, train_loader, val_dataset, val_loader
