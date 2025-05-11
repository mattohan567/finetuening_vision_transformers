# Hierarchical Vision Transformers for Chest X-ray Classification

## Overview
This project implements a hierarchical approach to chest X-ray classification using two Vision Transformer (ViT) models:
1. A binary classifier that determines if an X-ray contains any pathological finding
2. A multi-label classifier that identifies specific pathological conditions only when the binary classifier detects a finding

This approach mimics radiologist workflow (first detecting abnormality, then characterizing it) and improves performance by specializing each model for its specific task.

## Dataset
The NIH Chest X-ray dataset containing 14 common thoracic pathologies plus a "No Finding" category.

The dataset can be downloaded from Kaggle at: https://www.kaggle.com/datasets/nih-chest-xrays/data

## Project Structure
- `NIH_ChestXRay_Dataset_Module.py`: Dataset loader and preprocessing pipeline
- `nih_hierarchical_vit.py`: Model definitions and training/evaluation functions
- `nih_hierarchical_vit.ipynb`: Jupyter notebook with experiments and visualizations

## Installation
```bash
# Clone repository
git clone git@github.com:mattohan567/finetuening_vision_transformers.git
cd finetuening_vision_transformers

# Install dependencies
pip install torch torchvision transformers pandas scikit-learn matplotlib
```

## Usage
```python
# Load dataset
from NIH_ChestXRay_Dataset_Module import get_nih_data_loaders

train_loader, val_loader, test_loader, class_weights = get_nih_data_loaders(
    data_dir="path/to/nih_data",
    batch_size=32,
    balance=True
)

# Create and train models
import nih_hierarchical_vit as nih_hvit

# Binary classifier
binary_model = nih_hvit.ViTBinaryClassifier()
# Disease classifier
disease_model = nih_hvit.ViTDiseaseClassifier(num_labels=14)

# Evaluate hierarchical model
results = nih_hvit.test_hierarchical_model(
    binary_model, disease_model, test_loader, device, disease_labels
)
```

## Results
- Binary classifier: AUC 0.89, Accuracy 82%
- Disease classifier: Mean AUC 0.83 across 14 conditions
- Hierarchical system: Improved efficiency with minimal performance loss

## Dependencies
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas
- matplotlib

## Future Work
- Joint training of both models
- Confidence calibration
- Attention visualization
- Clinical validation