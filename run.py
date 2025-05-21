import torch
import argparse

from src.defect_detection.model import create_defect_detection_model
from src.defect_detection.train import train_defect_detection_model
from src.defect_detection.dataset import create_defect_detection_train_val_datasets_and_loaders
from src.defect_detection.evaluate import check_defect_detection_performance
from src.denoising.model import create_noise_type_detcetion_model, create_denoising_model
from src.denoising.dataset import create_noise_type_detcetion_train_val_datasets_and_loaders, create_denoising_train_val_datasets_and_loaders
from src.denoising.train import train_noise_type_detcetion_model, train_denoising_model
from src.denoising.evaluate import check_noise_type_detection_performance, check_denoising_performance
from src.denoising.denoise import denoise_datset
from src.config import ConfigManager


def run_phase1(device):
    print("Phase 1")
    model = create_defect_detection_model(device=device)

    dataset, train_dataset, train_loader, val_dataset, val_loader = create_defect_detection_train_val_datasets_and_loaders()
    model = train_defect_detection_model(model, train_loader, val_loader, device)

    print("Train scores:")
    check_defect_detection_performance(train_loader, model, split="train", device=device)
    print("validation scores:")
    check_defect_detection_performance(val_loader, model, split="validation", device=device)


def run_phase2_1(device):
    print("Phase 2.1")
    train_dataset, train_loader, val_dataset, val_loader = create_noise_type_detcetion_train_val_datasets_and_loaders()

    model = create_noise_type_detcetion_model(device)
    model = train_noise_type_detcetion_model(model, train_loader, val_loader, device)

    print("Train scores:")
    check_noise_type_detection_performance(train_loader, model, split="train", device=device)
    print("validation scores:")
    check_noise_type_detection_performance(val_loader, model, split="validation", device=device)


def run_phase2_2(device):
    print("Phase 2.2")
    for noise_type in ("Gaussian", "Periodic", "Salt"):
        train_dataset, train_loader, val_dataset, val_loader = create_denoising_train_val_datasets_and_loaders(noise_type=noise_type)

        model = create_denoising_model(device)
        model = train_denoising_model(model, train_loader, val_loader, noise_type=noise_type, device=device)

        print(f"{noise_type} train scores:")
        check_denoising_performance(model, train_loader, noise_type=noise_type, split="train", device=device)
        print(f"{noise_type} validation scores:")
        check_denoising_performance(model, val_loader, noise_type=noise_type, split="validation", device=device)


def run_phase3_1(device):
    print("Phase 3.1")
    denoise_datset(device)


def run_phase3_2(device):
    print("Phase 3.2")
    dataset, train_dataset, train_loader, val_dataset, val_loader = create_defect_detection_train_val_datasets_and_loaders()
    model = create_defect_detection_model(device)
    model = train_defect_detection_model(model, train_loader, val_loader, device)
    print("Train scores:")
    check_defect_detection_performance(train_loader, model, split="train", device=device)
    print("validation scores:")
    check_defect_detection_performance(val_loader, model, split="validation", device=device)


def parse_args():
    parser = argparse.ArgumentParser(description="Run specific phase of the denoising framework")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument('--phase', choices=['phase1', 'phase2.1', 'phase2.2', 'phase3.1', 'phase3.2'], help='Phase to run')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_path = args.config
    config = ConfigManager(config_path)  # Initialize the singleton with the config file

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.phase == 'phase1':
        run_phase1(device)
    elif args.phase == 'phase2.1':
        run_phase2_1(device)
    elif args.phase == 'phase2.2':
        run_phase2_2(device)
    elif args.phase == 'phase3.1':
        run_phase3_1(device)
    elif args.phase == 'phase3.2':
        run_phase3_2(device)

if __name__ == "__main__":
    main()
