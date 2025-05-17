import torch
import torchvision

from src.defect_detection.config import device


def create_defect_detection_model():
    model = torchvision.models.efficientnet_v2_l(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = torch.nn.Identity()
    model.classifier = torch.nn.Linear(in_features=327680, out_features=1)
    model.to(device)
    return model
