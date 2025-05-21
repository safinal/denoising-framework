import torch
import torchvision


def create_defect_detection_model(device):
    model = torchvision.models.efficientnet_v2_l(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = torch.nn.Identity()
    model.classifier = torch.nn.Linear(in_features=327680, out_features=1)
    model.to(device)
    return model
