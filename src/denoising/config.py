import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

denoising_in_channels = 1
learning_rate = 0.001
denoising_batch_size = 16
denoising_num_epochs = 30
denoising_val_fraction = 0.2
denoising_learning_rate = 0.001
denoising_logs_dir = "logs/denoising"

# noise type detection
num_classes = 3
learning_rate = 1e-3
noise_type_detection_batch_size = 256
noise_type_detcetion_num_epochs = 5
noise_type_detcetion_logs_dir = "logs/noise_type_detcetion"
noise_type_detcetion_val_fraction = 0.2
noise_type_detcetion_learning_rate = 0.001
load_model = False
save_model = False

denoised_dataset_root_dir = "DataSet1_Noise_Removed"