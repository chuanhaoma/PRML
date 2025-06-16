# POLLEN73S Image Classification Project

## 📝 Task Summary

This project implements image classification on the **POLLEN73S dataset** (73-class pollen grain classification) using deep learning models. The key components include:

- **Models Implemented**:
  - Vision Transformer (ViT-B/16)
  - DenseNet-201
  
- **Training Features**:
  - K-Fold cross-validation support (5-fold by default)
  - Learning rate scheduling with cosine annealing
  - Warmup learning rate scheduler
  - Model checkpoint saving
  - Comprehensive training metrics tracking

- **Performance**:
  - Achieves up to **98.85% accuracy** with DenseNet-201
  - Vision Transformer achieves **99.01% accuracy**

## 🚀 How to Run

### 🔧 1. Installation
```bash
# Access dataset at
# https://figshare.com/articles/dataset/POLLEN73S/12536573?file=23307950

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scikit-learn opencv-python matplotlib seaborn tqdm
```

### 🤖 2. Training Models
1. Configure training parameters in `k_fold.py`:
```python
# Example: Run DenseNet-201 with 5-fold cross-validation
if True:
    # Set training epochs to 75.
    EPOCHS = 75
    # Replace 'k_fold_val_DenseNet_A' with your own validation func.
    best_accs, mean, std, histories = k_fold_val_DenseNet_A(epochs=EPOCHS)
```

2. Execute training:
```bash
python k_fold.py
```

## 📋 Requirements

### 📦 1. Python Packages
| Package | Version |
|---------|---------|
| torch | >=1.10 |
| torchvision | >=0.11 |
| numpy | >=1.20 |
| scikit-learn | >=1.0 |
| opencv-python | >=4.5 |
| matplotlib | >=3.5 |
| seaborn | >=0.11 |
| tqdm | >=4.64 |

### 💻 2. Hardware Recommendations
- **GPU**: NVIDIA GPU with ≥8GB VRAM (CUDA 11.x compatible)
- **RAM**: ≥16GB system memory
- **Storage**: ≥10GB free space for dataset and models

## 📂 Project Structure
```
project/
├── code/                    # Project code
|    ├── dataset.py          # Dataset loading and preprocessing
|    ├── model.py            # Model architectures (ViT, DenseNet)
|    ├── train.py            # Training loop and utilities
|    ├── k_fold.py           # K-Fold cross-validation
|    ├── scheduler.py        # Learning rate schedulers
|    ├── figure.py           # Visualization functions
|    ├── result_visualize.py # Result visualization script
|    └── save.py             # Save model to file
├── dataset/                 # POLLEN73S dataset
├── figure/                  # Saved k-fold figures
└── model/                   # Saved model checkpoints
```

## 🏆 Best Performance Results
| Model | Accuracy | Standard Deviation | Epochs | Fold |
|-------|----------|---------------------|--------|-----|
| DenseNet-201 | 98.85% | ±0.38% | 170 (50 warmup + 120 cosine) | 5 |
| Vision Transformer | 99.01% | ±0.49% | 170 (50 warmup + 120 cosine) | 5 |
