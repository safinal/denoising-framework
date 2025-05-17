import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channels = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 256
num_epochs = 10
k_folds = 5

base_logs_dir = "logs/defect_detection"

dataset_root_dir = "DataSet1"

defect_detection_val_fraction = 0.2
defect_detection_learning_rate = 0.001