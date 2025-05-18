import torch
import torchmetrics
import os
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from src.defect_detection.config import device, base_logs_dir, k_folds, batch_size, learning_rate, num_epochs
from src.defect_detection.model import create_defect_detection_model


def check_defect_detection_performance(loader, model, experiment_logs_dir, split):
    model.eval()
    full_y = torch.tensor([], device=device, dtype=torch.int8)
    full_predictions = torch.tensor([], device=device)
    full_scores = torch.tensor([], device=device)

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x).reshape(-1)
            predictions = (scores >= 0).float()
            full_y = torch.cat(tensors=(full_y, y))
            full_predictions = torch.cat(tensors=(full_predictions, predictions))
            full_scores = torch.cat(tensors=(full_scores, scores))
            

        accuracy = torchmetrics.functional.classification.binary_accuracy(full_predictions, full_y)
        precision = torchmetrics.functional.classification.binary_precision(full_predictions, full_y)
        recall = torchmetrics.functional.classification.binary_recall(full_predictions, full_y)
        f1_score = torchmetrics.functional.classification.binary_f1_score(full_predictions, full_y)
        specificity = torchmetrics.functional.classification.binary_specificity(full_predictions, full_y)
        auroc = torchmetrics.functional.classification.binary_auroc(full_scores, full_y)

    results = {
        "accuracy": accuracy.detach().cpu().item(), 
        "precision": precision.detach().cpu().item(), 
        "recall": recall.detach().cpu().item(), 
        "f1_score": f1_score.detach().cpu().item(), 
        "specificity": specificity.detach().cpu().item(), 
        "auroc": auroc.detach().cpu().item()
    }
    print(results)
    os.makedirs(os.path.join(base_logs_dir, experiment_logs_dir), exist_ok=True)
    with open(os.path.join(base_logs_dir, experiment_logs_dir, f"{split}_results.json"), 'w') as f:
        json.dump(results, f)

    metric = torchmetrics.classification.BinaryROC()
    metric.update(full_scores, full_y)
    fig, ax = metric.plot(score=True)
    fig.savefig(os.path.join(base_logs_dir, experiment_logs_dir, f"{split}_roc_curve.png"))
    plt.close(fig)

    metric = torchmetrics.classification.BinaryConfusionMatrix().to(device)
    metric.update(full_predictions, full_y)
    fig, ax = metric.plot()
    fig.savefig(os.path.join(base_logs_dir, experiment_logs_dir, f"{split}_confusion_matrix.png"))
    plt.close(fig)


def defect_detection_k_fold_cross_validation(dataset, criterion, experiment_logs_dir):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

        model = create_defect_detection_model()
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        model.train()
        for epoch in range(num_epochs):
            loop = tqdm(train_loader, total=len(train_loader), leave=True)
            for data, targets in loop:
                data = data.to(device=device)
                targets = targets.to(device=device)

                scores = model(data).reshape(-1)
                loss = criterion(scores, targets.float())

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
                loop.set_postfix(loss=loss.item())

        print(f"Val set: ", end='')
        check_defect_detection_performance(val_loader, model, experiment_logs_dir=experiment_logs_dir, split=f"cross_val_{fold}")
