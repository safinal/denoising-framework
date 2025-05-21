import torch
import torchvision

from src.config import ConfigManager


class DenoisingAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(DenoisingAutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),

            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),

            torch.nn.Conv2d(in_channels=128, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def create_noise_type_detcetion_model(device, checkpoint_path=None):
    model = torchvision.models.efficientnet_v2_l(weights='DEFAULT' if checkpoint_path is None else None)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = torch.nn.Identity()
    model.classifier[1] = torch.nn.Linear(in_features=327680, out_features=3, bias=True)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.to(device)
    return model

def create_denoising_model(device, checkpoint_path=None):
    model = DenoisingAutoEncoder(in_channels=ConfigManager().get("in_channels"))
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.to(device)
    return model
