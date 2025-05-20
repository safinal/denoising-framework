# Image Denoising and Defect Detection Framework

This framework implements a multi-phase approach to image denoising and defect detection, combining deep learning techniques to handle different types of noise and improve defect detection accuracy.


## Features

- **Phase 1**: Defect Detection
  - Binary classification model for defect detection
  - Uses BCEWithLogitsLoss for training
  - AdamW optimizer with configurable learning rate

- **Phase 2.1**: Noise Type Detection
  - Multi-class classification for identifying noise types
  - Supports Gaussian, Periodic, and Salt noise types
  - CrossEntropyLoss for training

- **Phase 2.2**: Denoising Models
  - Separate denoising models for each noise type
  - Custom architecture for effective noise removal
  - Individual training pipelines for each noise type

- **Phase 3**: End-to-End Pipeline
  - Combines noise type detection and denoising
  - Processes entire dataset with appropriate denoising
  - Retrains defect detection on denoised images

## Requirements

- Python 3.x
- PyTorch
- torchvision
- torchmetrics
- scikit-learn
- numpy
- pandas
- matplotlib
- tqdm
- opencv-python
- gdown
- openpyxl

## Installation

1. Clone the repository:
```bash
git clone https://github.com/safinal/denoising-framework.git
cd denoising-framework
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The framework can be run in different phases using the main script:

```bash
python run.py [phase]
```

Available phases:
- `phase1`: Run defect detection training
- `phase2.1`: Train noise type detection model
- `phase2.2`: Train denoising models for each noise type
- `phase3`: Run the complete pipeline (noise detection + denoising + defect detection)

## Configuration

The framework uses configuration parameters defined in:
- `src/denoising/config.py`: Denoising and noise type detection parameters
- `src/defect_detection/config.py`: Defect detection parameters


## License

This project is licensed under the terms of the included LICENSE file.
