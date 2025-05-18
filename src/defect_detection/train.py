import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from src.defect_detection.config import num_epochs, device, base_logs_dir


def train_defect_detection_model(model, train_loader, val_loader, optimizer, criterion, experiment_logs_dir):
    train_f1_list = []
    validation_f1_list = []

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        all_preds = []
        all_targets = []
        for data, targets in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data).reshape(-1)
            loss = criterion(scores, targets.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (scores >= 0).float().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), train_f1=f1_score(all_targets, all_preds))
        train_f1_list.append(f1_score(all_targets, all_preds))

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = model(data).reshape(-1)
                preds = (scores >= 0).float().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
        val_f1 = f1_score(all_targets, all_preds)
        validation_f1_list.append(val_f1)
        print(f"Validation F1 Score: {val_f1}")
        if validation_f1_list[-1] == max(validation_f1_list):
            os.makedirs(os.path.join(base_logs_dir, experiment_logs_dir), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(base_logs_dir, experiment_logs_dir, "best_model.pth"))

    plt.plot(train_f1_list, label="Train")
    plt.plot(validation_f1_list, label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(base_logs_dir, experiment_logs_dir, "train_val_f1_plot.png"))
    plt.close()

    model.load_state_dict(torch.load(os.path.join(base_logs_dir, experiment_logs_dir, "best_model.pth"), weights_only=True))
    return model
