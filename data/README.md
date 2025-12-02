# 📊 Dataset Information

## Overview

The dataset focuses on three common rice diseases: **Bacterial Blight**, **Brown Spot**, and **Rice Blast**.

## Dataset Access

The processed and combined dataset is available on Hugging Face:

- **Hugging Face Dataset**: [paddy-disease-classification](https://huggingface.co/datasets/hello-wald/paddy-disease-classification)
  - **Format**: ImageFolder
  - **Total Images**: 2,861 (2,495 train, 366 test)

## Data Sources

### 1. Kaggle Dataset: 20k Multi-Class Crop Disease Images

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images)
- **Description**: A large-scale dataset containing 20,000+ images of various crop diseases

### 2. Paddy Doctor Dataset

- **Source**: [Paddy Doctor Dataset](https://paddydoc.github.io/dataset/)
- **Description**: A specialized dataset for paddy disease and pest detection

## Selected Classes

From the combined datasets, only three disease classes were selected for this project:

| Disease | Description |
|---------|-------------|
| **Bacterial Blight** | Caused by *Xanthomonas oryzae*, characterized by water-soaked lesions that turn yellow and then brown |
| **Brown Spot** | Fungal disease (*Cochliobolus miyabeanus*) causing brown oval spots on leaves |
| **Rice Blast** | Caused by *Magnaporthe oryzae*, produces diamond-shaped lesions with gray centers |

## Dataset Structure

```
data/processed/
├── train/
│   ├── Bacterial Blight/    # ~631 images
│   ├── Brown Spot/          # ~944 images
│   └── Rice Blast/          # ~920 images
└── test/
    ├── Bacterial Blight/    # Test images
    ├── Brown Spot/          # Test images
    └── Rice Blast/         # ~120 images
```

## Dataset Statistics

### Training Set

- **Bacterial Blight**: 631 images
- **Brown Spot**: 944 images
- **Rice Blast**: 920 images
- **Total Training Images**: 2,495 images

### Test Set

- **Bacterial Blight**: ~134 images
- **Brown Spot**: ~112 images
- **Rice Blast**: ~120 images
- **Total Test Images**: 366 images

## Image Characteristics

- **Format**: JPG/JPEG
- **Resolution**: Variable (original sources had different resolutions)
- **Color Space**: RGB

## Usage

### Local Directory Structure

The dataset is organized in a class-based directory structure suitable for PyTorch's `ImageFolder` or custom dataset loaders. The `RiceDiseaseDataset` class in `src/data_loader.py` handles loading and preprocessing.

### Hugging Face Dataset
You can also load the dataset directly from Hugging Face:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("hello-wald/paddy-disease-classification")

# Access train and test splits
train_data = dataset["train"]
test_data = dataset["test"]
```

## Citation

If you use this dataset, please cite the original sources:

1. **Kaggle Dataset**: [20k Multi-Class Crop Disease Images](https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images)

2. **Paddy Doctor Dataset**:
   - Website: [Paddy Doctor](https://paddydoc.github.io/dataset/)
   - Please refer to the original publication for proper citation

3. **Processed Dataset**:
   - Hugging Face: [hello-wald/paddy-disease-classification](https://huggingface.co/datasets/hello-wald/paddy-disease-classification)