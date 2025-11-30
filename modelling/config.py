"""Configuration parameters for rice disease classification"""

import torch

# Path configuration
TRAIN_DATA_DIR = './dataset/train'
TEST_DATA_DIR = './dataset/test'
MODEL_SAVE_PATH = 'best_rice_disease_model.pth'
GRAPH_DIR = 'graphs'

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 5

# Model parameters
NUM_CLASSES = 3
MODEL_NAME = 'efficientnet_b4'
CLASS_NAMES = ['Bacterial Blight', 'Brown Spot', 'Rice Blast']
CLASSIFIER_UNITS = 512
ACTIVATION_FUNCTION = 'gelu'  # Options: 'relu', 'gelu'

# Hardware configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

# Focal Loss parameters
FOCAL_LOSS_ALPHA = 1
FOCAL_LOSS_GAMMA = 2