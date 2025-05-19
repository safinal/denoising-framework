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
from src.denoising.config import noise_type_detcetion_val_fraction, noise_type_detcetion_learning_rate, denoising_val_fraction, denoising_learning_rate, denoised_dataset_root_dir
from src.defect_detection.config import defect_detection_val_fraction, defect_detection_learning_rate, dataset_root_dir


def run_phase1():
    print("Phase 1")
    experiment_logs_dir = "phase 1"
    defect_detection_model = create_defect_detection_model()

    defect_detection_dataset, defect_detection_train_dataset, defect_detection_train_loader, defect_detection_val_dataset, defect_detection_val_loader = create_defect_detection_train_val_datasets_and_loaders(dataset_root_dir=dataset_root_dir, val_fraction=defect_detection_val_fraction)

    defect_detection_criterion = torch.nn.BCEWithLogitsLoss()
    defect_detection_optimizer = torch.optim.AdamW(params=defect_detection_model.parameters(), lr=defect_detection_learning_rate)

    defect_detection_model = train_defect_detection_model(defect_detection_model, defect_detection_train_loader, defect_detection_val_loader, defect_detection_optimizer, defect_detection_criterion, experiment_logs_dir=experiment_logs_dir)

    print("Train scores:")
    check_defect_detection_performance(defect_detection_train_loader, defect_detection_model, experiment_logs_dir=experiment_logs_dir, split="train")
    print("validation scores:")
    check_defect_detection_performance(defect_detection_val_loader, defect_detection_model, experiment_logs_dir=experiment_logs_dir, split="validation")

def run_phase2_1():
    print("Phase 2.1")
    noise_type_detcetion_train_dataset, noise_type_detcetion_train_loader, noise_type_detcetion_val_dataset, noise_type_detcetion_val_loader = create_noise_type_detcetion_train_val_datasets_and_loaders(val_fraction=noise_type_detcetion_val_fraction)

    noise_type_detcetion_model = create_noise_type_detcetion_model()
    noise_type_detcetion_criterion = torch.nn.CrossEntropyLoss()
    noise_type_detcetion_optimizer = torch.optim.AdamW(params=noise_type_detcetion_model.parameters(), lr=noise_type_detcetion_learning_rate)

    noise_type_detcetion_model = train_noise_type_detcetion_model(noise_type_detcetion_model, noise_type_detcetion_train_loader, noise_type_detcetion_val_loader, noise_type_detcetion_optimizer, noise_type_detcetion_criterion)

    print("Train scores:")
    check_noise_type_detection_performance(noise_type_detcetion_train_loader, noise_type_detcetion_model, split="train")
    print("validation scores:")
    check_noise_type_detection_performance(noise_type_detcetion_val_loader, noise_type_detcetion_model, split="validation")

def run_phase2_2():
    print("Phase 2.2")
    gaussian_denoising_train_dataset, gaussian_denoising_train_loader, gaussian_denoising_val_dataset, gaussian_denoising_val_loader = create_denoising_train_val_datasets_and_loaders(noise_type="Gaussian", val_fraction=denoising_val_fraction)
    periodic_denoising_train_dataset, periodic_denoising_train_loader, periodic_denoising_val_dataset, periodic_denoising_val_loader = create_denoising_train_val_datasets_and_loaders(noise_type="Periodic", val_fraction=denoising_val_fraction)
    salt_denoising_train_dataset, salt_denoising_train_loader, salt_denoising_val_dataset, salt_denoising_val_loader = create_denoising_train_val_datasets_and_loaders(noise_type="Salt", val_fraction=denoising_val_fraction)

    gaussian_denoising_model = create_denoising_model()
    gaussian_denoising_criterion = torch.nn.MSELoss()
    gaussian_denoising_optimizer = torch.optim.AdamW(params=gaussian_denoising_model.parameters(), lr=denoising_learning_rate)
    gaussian_denoising_model = train_denoising_model(gaussian_denoising_model, gaussian_denoising_train_loader, gaussian_denoising_val_loader, gaussian_denoising_optimizer, gaussian_denoising_criterion, noise_type="gaussian")

    print("Train scores:")
    check_denoising_performance(gaussian_denoising_model, gaussian_denoising_train_loader, noise_type="gaussian", split="train")
    print("validation scores:")
    check_denoising_performance(gaussian_denoising_model, gaussian_denoising_val_loader, noise_type="gaussian", split="validation")

    periodic_denoising_model = create_denoising_model()
    periodic_denoising_criterion = torch.nn.MSELoss()
    periodic_denoising_optimizer = torch.optim.AdamW(params=periodic_denoising_model.parameters(), lr=denoising_learning_rate)
    periodic_denoising_model = train_denoising_model(periodic_denoising_model, periodic_denoising_train_loader, periodic_denoising_val_loader, periodic_denoising_optimizer, periodic_denoising_criterion, noise_type="periodic")

    print("Train scores:")
    check_denoising_performance(periodic_denoising_model, periodic_denoising_train_loader, noise_type="periodic", split="train")
    print("validation scores:")
    check_denoising_performance(periodic_denoising_model, periodic_denoising_val_loader, noise_type="periodic", split="validation")

    salt_denoising_model = create_denoising_model()
    salt_denoising_criterion = torch.nn.MSELoss()
    salt_denoising_optimizer = torch.optim.AdamW(params=salt_denoising_model.parameters(), lr=denoising_learning_rate)
    salt_denoising_model = train_denoising_model(salt_denoising_model, salt_denoising_train_loader, salt_denoising_val_loader, salt_denoising_optimizer, salt_denoising_criterion, noise_type="salt")

    print("Train scores:")
    check_denoising_performance(salt_denoising_model, salt_denoising_train_loader, noise_type="salt", split="train")
    print("validation scores:")
    check_denoising_performance(salt_denoising_model, salt_denoising_val_loader, noise_type="salt", split="validation")

def run_phase3():
    print("Phase 3")
    experiment_logs_dir = "phase 3"
    noise_type_detcetion_model = create_noise_type_detcetion_model()
    noise_type_detcetion_model.load_state_dict(torch.load("logs/noise_type_detcetion/best_model.pth", weights_only=True))
    gaussian_denoising_model = create_denoising_model()
    gaussian_denoising_model.load_state_dict(torch.load("logs/denoising/gaussian/best_model.pth", weights_only=True))
    periodic_denoising_model = create_denoising_model()
    periodic_denoising_model.load_state_dict(torch.load("logs/denoising/periodic/best_model.pth", weights_only=True))
    salt_denoising_model = create_denoising_model()
    salt_denoising_model.load_state_dict(torch.load("logs/denoising/salt/best_model.pth", weights_only=True))
    denoise_datset(
        noise_classifier=noise_type_detcetion_model, 
        gaussian_denoiser=gaussian_denoising_model, 
        periodic_denoiser=periodic_denoising_model, 
        salt_denoiser=salt_denoising_model,
        denoised_dataset_root_dir=denoised_dataset_root_dir
    )

    defect_detection_dataset, defect_detection_train_dataset, defect_detection_train_loader, defect_detection_val_dataset, defect_detection_val_loader = create_defect_detection_train_val_datasets_and_loaders(dataset_root_dir=denoised_dataset_root_dir, val_fraction=defect_detection_val_fraction)
    defect_detection_model = create_defect_detection_model()
    defect_detection_criterion = torch.nn.BCEWithLogitsLoss()
    defect_detection_optimizer = torch.optim.AdamW(params=defect_detection_model.parameters(), lr=defect_detection_learning_rate)
    defect_detection_model = train_defect_detection_model(defect_detection_model, defect_detection_train_loader, defect_detection_val_loader, defect_detection_optimizer, defect_detection_criterion, experiment_logs_dir=experiment_logs_dir)
    print("Train scores:")
    check_defect_detection_performance(defect_detection_train_loader, defect_detection_model, experiment_logs_dir=experiment_logs_dir, split="train")
    print("validation scores:")
    check_defect_detection_performance(defect_detection_val_loader, defect_detection_model, experiment_logs_dir=experiment_logs_dir, split="validation")

def main():
    parser = argparse.ArgumentParser(description='Run specific phase of the denoising framework')
    parser.add_argument('phase', choices=['phase1', 'phase2.1', 'phase2.2', 'phase3'], help='Phase to run')
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.phase == 'phase1':
        run_phase1()
    elif args.phase == 'phase2.1':
        run_phase2_1()
    elif args.phase == 'phase2.2':
        run_phase2_2()
    elif args.phase == 'phase3':
        run_phase3()

if __name__ == '__main__':
    main()
