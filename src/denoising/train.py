import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from src.denoising.config import denoising_num_epochs, noise_type_detcetion_num_epochs, device, denoising_logs_dir, noise_type_detcetion_logs_dir


def train_denoising_model(model, train_loader, val_loader, optimizer, criterion, noise_type):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(denoising_num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = model(data)
                loss = criterion(scores, targets)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.join(denoising_logs_dir, noise_type), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(denoising_logs_dir, noise_type, "best_model.pth"))
    
    # Plot and save both train and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(denoising_logs_dir, "loss_plot.png"))
    plt.close()

    # Load best model
    model.load_state_dict(torch.load(os.path.join(denoising_logs_dir, noise_type, "best_model.pth"), weights_only=True))
    return model
        


def train_noise_type_detcetion_model(model, train_loader, val_loader, optimizer, criterion):
    train_f1_list = []
    validation_f1_list = []

    for epoch in range(noise_type_detcetion_num_epochs):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        all_predictions = []
        all_targets = []
        for data, targets in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For multi-class, we use argmax to get predicted class
            predictions = scores.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Calculate F1 score for current batch
            batch_f1 = f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='macro')
            loop.set_description(f"Epoch [{epoch+1}/{noise_type_detcetion_num_epochs}]")
            loop.set_postfix(loss=loss.item(), train_f1=batch_f1)
        
        # Calculate overall F1 score for training
        train_f1 = f1_score(all_targets, all_predictions, average='macro')
        train_f1_list.append(train_f1)

        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = model(data)
                predictions = scores.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate F1 score for validation
        validation_f1 = f1_score(all_targets, all_predictions, average='macro')
        validation_f1_list.append(validation_f1)
        print(f"Validation F1 Score: {validation_f1}")
        if validation_f1 == max(validation_f1_list):
            os.makedirs(noise_type_detcetion_logs_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(noise_type_detcetion_logs_dir, "best_model.pth"))

    plt.plot(train_f1_list, label="Train")
    plt.plot(validation_f1_list, label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(noise_type_detcetion_logs_dir, "train_val_f1_plot.png"))
    plt.close()

    model.load_state_dict(torch.load(os.path.join(noise_type_detcetion_logs_dir, "best_model.pth"), weights_only=True))
    return model
