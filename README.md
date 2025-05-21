# Denoising Framework

A robust framework for image denoising and defect detection, leveraging a multi-phase deep learning approach to effectively handle various noise types and enhance defect detection accuracy.

## Features

### Phase 1: Defect Detection
- Implements a binary classification model for accurate defect detection
- Utilizes BCEWithLogitsLoss for model training
- Employs AdamW optimizer with configurable learning rate

### Phase 2.1: Noise Type Detection
- Performs multi-class classification to identify noise types
- Supports Gaussian, Periodic, and Salt noise types
- Uses CrossEntropyLoss for training

### Phase 2.2: Denoising Models
- Provides dedicated denoising models for each noise type
- Features a custom architecture optimized for noise removal
- Includes individual training pipelines for each noise type

### Phase 3.1: Dataset Denoising
- Applies trained models to denoise the entire dataset
- Integrates noise type detection and denoising for seamless processing

### Phase 3.2: Defect Detection on Denoised Data
- Retrains the defect detection model using denoised images

## Installation

Follow these steps to set up the framework:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/safinal/denoising-framework.git
   cd denoising-framework
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the framework in different phases using the main script:

```bash
python run.py --phase [PHASE] --config [YAML_CONFIG_PATH]
```

### Available Phases
- `phase1`: Trains the defect detection model
- `phase2.1`: Trains the noise type detection model
- `phase2.2`: Trains denoising models for each noise type
- `phase3.1`: Denoises the dataset using trained models
- `phase3.2`: Retrains defect detection on the denoised dataset

### Configuration
The framework uses YAML configuration files located in the `src/config/` directory to specify parameters.

## License

This project is licensed under the terms of the included LICENSE file.
