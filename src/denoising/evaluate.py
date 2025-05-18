import torch
import torchmetrics
import os
import json
from matplotlib import pyplot as plt

from src.denoising.config import device, noise_type_detcetion_logs_dir, denoising_logs_dir


def check_denoising_performance(model, loader, noise_type, split):
    model.eval()
    psnr = 0
    lpips = 0
    ssim = 0
    count = 0
    with torch.no_grad():
        for noisy_image_gray, noise_free_image_gray in loader:
            noisy_image_gray = noisy_image_gray.to(device)
            noise_free_image_gray = noise_free_image_gray.to(device)

            output = model(noisy_image_gray)
            
            lpips += noisy_image_gray.shape[0] * torchmetrics.functional.image.learned_perceptual_image_patch_similarity(noise_free_image_gray.repeat(1, 3, 1, 1), output.repeat(1, 3, 1, 1), net_type='alex', normalize=True)
            psnr += noisy_image_gray.shape[0] * torchmetrics.functional.image.peak_signal_noise_ratio(output, noise_free_image_gray, data_range=(0, 1))
            ssim += noisy_image_gray.shape[0] * torchmetrics.functional.image.structural_similarity_index_measure(output, noise_free_image_gray, data_range=(0, 1))
            count += noisy_image_gray.shape[0]
    psnr, ssim, lpips = psnr / count, ssim / count, lpips / count
    results = {
        "psnr": psnr.detach().cpu().numpy(), 
        "ssim": ssim.detach().cpu().numpy(), 
        "lpips": lpips.detach().cpu().numpy()
    }
    print(results)
    os.makedirs(os.path.join(denoising_logs_dir, noise_type), exist_ok=True)
    with open(os.path.join(denoising_logs_dir, noise_type, f"{split}_results.json"), 'w') as f:
        json.dump(results, f)



def check_noise_type_detection_performance(loader, model, split):
    model.eval()
    full_y = torch.tensor([], device=device, dtype=torch.int8)
    full_predictions = torch.tensor([], device=device)
    full_scores = torch.tensor([], device=device)

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            predictions = scores.max(1)[1]
            full_y = torch.cat(tensors=(full_y, y))
            full_predictions = torch.cat(tensors=(full_predictions, predictions))
            full_scores = torch.cat(tensors=(full_scores, scores))

        accuracy = torchmetrics.functional.classification.multiclass_accuracy(full_predictions, full_y, num_classes=3, average='macro')
        precision = torchmetrics.functional.classification.multiclass_precision(full_predictions, full_y, num_classes=3, average='macro')
        recall = torchmetrics.functional.classification.multiclass_recall(full_predictions, full_y, num_classes=3, average='macro')
        f1_score = torchmetrics.functional.classification.multiclass_f1_score(full_predictions, full_y, num_classes=3, average='macro')
        specificity = torchmetrics.functional.classification.multiclass_specificity(full_predictions, full_y, num_classes=3, average='macro')
        auroc = torchmetrics.functional.classification.multiclass_auroc(full_scores, full_y, num_classes=3, average='macro')
        # fpr, tpr, thresholds = torchmetrics.functional.classification.multiclass_roc(full_scores, full_y, num_classes=3)
    
    results = {
        "accuracy": accuracy.detach().cpu().item(), 
        "precision": precision.detach().cpu().item(), 
        "recall": recall.detach().cpu().item(), 
        "f1_score": f1_score.detach().cpu().item(), 
        "specificity": specificity.detach().cpu().item(), 
        "auroc": auroc.detach().cpu().item(), 
    }
    print(results)
    os.makedirs(noise_type_detcetion_logs_dir, exist_ok=True)
    with open(os.path.join(noise_type_detcetion_logs_dir, f"{split}_results.json"), 'w') as f:
        json.dump(results, f)
        
    metric = torchmetrics.classification.MulticlassROC(num_classes=3)
    metric.update(full_scores, full_y)
    fig, ax = metric.plot(score=True)
    fig.savefig(os.path.join(noise_type_detcetion_logs_dir, f"{split}_roc_curve.png"))
    plt.close(fig)

    metric = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=3).to(device)
    metric.update(full_predictions, full_y)
    fig, ax = metric.plot()
    fig.savefig(os.path.join(noise_type_detcetion_logs_dir, f"{split}_confusion_matrix.png"))
    plt.close(fig)
