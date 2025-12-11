"""Configuration parameters for rice disease classification"""

import torch
import platform
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Path configuration
TRAIN_DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'train'
TEST_DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'test'
MODEL_SAVE_PATH = PROJECT_ROOT / 'outputs' / 'models' / 'best_rice_disease_model.pth'
GRAPH_DIR = PROJECT_ROOT / 'outputs' / 'plots'

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
CLASSIFIER_UNITS = 256
ACTIVATION_FUNCTION = 'gelu'  # Options: 'relu', 'gelu'

# Hardware configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

if platform.system() == 'Darwin':  # macOS
    NUM_WORKERS = 0
    USE_PERSISTENT_WORKERS = False
else:
    NUM_WORKERS = 4
    USE_PERSISTENT_WORKERS = True

# Focal Loss parameters
FOCAL_LOSS_ALPHA = 1
FOCAL_LOSS_GAMMA = 2