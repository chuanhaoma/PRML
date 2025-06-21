# POLLEN73S Image Classification Project

## 📝 Introduction

This project implements image classification on the **POLLEN73S dataset** (73-class pollen grain classification) using deep learning models. The key components include:

- **Models Implemented**:
  - ResNet-50
  - DenseNet-201
  - Vision Transformer (ViT-B-16, ViT-L-16, ViT-H-14)
  
- **Training Features**:
  - K-Fold cross-validation support (5-fold by default)
  - Learning rate scheduler with cosine decay and warmup
  - Model checkpoint saving
  - Comprehensive training metrics tracking
  - Model evaluation

- **Performance**:
  - Achieves up to **98.85% accuracy** with DenseNet-201
  - ViT-B-16 achieves **99.01% accuracy**

## 🚀 How to Run

### 🔧 1. Installation
```bash
# Access the POLLEN73S dataset at
# https://figshare.com/articles/dataset/POLLEN73S/12536573?file=23307950

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scikit-learn opencv-python matplotlib seaborn tqdm
```

### 🤖 2. Training Models
1. Configure training parameters in `k_fold.py`:
```python
# Example: Run ViT-B-16 with 5-fold cross-validation
if True:
    WARMUP = 50
    COSINE = 120
    model, best_accs, mean, std, histories = k_fold_val(model_class=ViTForPollenClassification, warmup_steps=WARMUP, cosine_steps=COSINE, save_model=True, prefix='vitb')
    save_report('./model/vit_report.txt', best_accs, mean, std, COSINE)
    evaluate_deep(model, model_name='ViT-B-16')
    for i in range(len(histories)):
        save_figure(histories[i], f"./figure/vit_{i:02d}.png", f"Fine-tuning on ViT-B-16 Model - Fold {i} Epoch {COSINE}")
```

2. Make sure you have established all necessary folders
```bash
mkdir evaluate/ figure/ model/
```

3. Execute training and k-fold validation:
```bash
python3 code/k_fold.py | tee -a log.log
```

## 📋 Requirements
### 🐍 0. Python Version
| Python | Version |
|---------|---------|
| Recommended | 3.10 |
| Also Available | 3.11.6 |

### 📦 1. Python Packages
| Package | Version |
|---------|---------|
| torch | 2.7.1 |
| torchvision | 0.21.0 |
| numpy | 1.26.4 |
| scikit-learn | 1.7.0 |
| opencv-python | 4.11.0 |
| matplotlib | 3.10.3 |
| seaborn | 0.13.2 |
| tqdm | 4.66.4 |

### 💻 2. Hardware Recommendations
- **GPU**: NVIDIA GPU with ≥8GB VRAM (CUDA 11.x compatible)
- **RAM**: ≥16GB system memory
- **Storage**: ≥30GB free space for dataset and models

## 📂 Project Structure
```
project/
├── code/                    # Project code
|    ├── dataset.py          # Dataset loading and preprocessing
|    ├── evaluate.py         # Evaluate the model by calculate the macro F1-socre and the macro AUROC
|    ├── k_fold.py           # Train models and eval K-Fold cross-validation
|    ├── model.py            # Model architectures (ViT, DenseNet)
|    ├── save.py             # Misc functions like saving/loading model
|    ├── scheduler.py        # Learning rate schedulers
|    └── train.py            # Training loop and utilities
├── dataset/                 # POLLEN73S datasets
├── evaluate/                # Evaluation results
├── figure/                  # Saved k-fold figures
└── model/                   # Saved model checkpoints
```

## 🏆 Performance Results
| Model | Accuracy | Standard Deviation | Epochs | Fold |
|-------|----------|---------------------|--------|-----|
| ResNet-50 | 98.18% | ±0.49% | 80 (30 warmup + 50 cosine) | 5 |
| DenseNet-201 | 98.85% | ±0.38% | 170 (50 warmup + 120 cosine) | 5 |
| ViT-B-16 | 99.01% | ±0.49% | 170 (50 warmup + 120 cosine) | 5 |
| ViT-L-16 | 98.77% | ±0.73% | 50 (25 warmup + 25 cosine) | 5 |