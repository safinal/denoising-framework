import torch
import numpy as np
import os
import cv2

from src.denoising.config import device


def denoise_datset(noise_classifier, gaussian_denoiser, periodic_denoiser, salt_denoiser, denoised_dataset_root_dir):
    os.mkdir(denoised_dataset_root_dir)
    os.mkdir(os.path.join(denoised_dataset_root_dir, 'Defected'))
    os.mkdir(os.path.join(denoised_dataset_root_dir, 'OK'))
    with torch.no_grad():
        for image_name in os.listdir(os.path.join('DataSet1', 'Defected')):
            noisy_image_gray = torch.from_numpy(cv2.imread(os.path.join('DataSet1', 'Defected', image_name), flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255).unsqueeze(0).unsqueeze(0).to(device)
            noisy_image_rgb = torch.from_numpy(cv2.imread(os.path.join('DataSet1', 'Defected', image_name), flags=cv2.IMREAD_COLOR).astype(np.float32) / 255).permute((2, 0, 1)).unsqueeze(0).to(device)
            scores = noise_classifier(noisy_image_rgb)
            predictions = scores.max(1)[1]
            if predictions[0] == 0: # Gaussian
                output = gaussian_denoiser(noisy_image_gray).squeeze(0).squeeze(0)
            elif predictions[0] == 1: # Periodic
                output = periodic_denoiser(noisy_image_gray).squeeze(0).squeeze(0)
            else: # Salt
                output = salt_denoiser(noisy_image_gray).squeeze(0).squeeze(0)

            cv2.imwrite(os.path.join(denoised_dataset_root_dir, 'Defected', f"{image_name}"), cv2.cvtColor(cv2.normalize(src=output.detach().cpu().numpy(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_GRAY2BGR))

        for image_name in os.listdir(os.path.join('DataSet1', 'OK')):
            noisy_image_gray = torch.from_numpy(cv2.imread(os.path.join('DataSet1', 'OK', image_name), flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255).unsqueeze(0).unsqueeze(0).to(device)
            noisy_image_rgb = torch.from_numpy(cv2.imread(os.path.join('DataSet1', 'OK', image_name), flags=cv2.IMREAD_COLOR).astype(np.float32) / 255).permute((2, 0, 1)).unsqueeze(0).to(device)
            scores = noise_classifier(noisy_image_rgb)
            predictions = scores.max(1)[1]
            if predictions[0] == 0: # Gaussian
                output = gaussian_denoiser(noisy_image_gray).squeeze(0).squeeze(0)
            elif predictions[0] == 1: # Periodic
                output = periodic_denoiser(noisy_image_gray).squeeze(0).squeeze(0)
            else: # Salt
                output = salt_denoiser(noisy_image_gray).squeeze(0).squeeze(0)

            cv2.imwrite(os.path.join(denoised_dataset_root_dir, 'OK', f"{image_name}"), cv2.cvtColor(cv2.normalize(src=output.detach().cpu().numpy(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_GRAY2BGR))
