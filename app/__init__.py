from src.model import Classifier
from src.data_loader import RiceDiseaseDataset
from config import config

def _lazy_import_train():
    from src.train import train_model
    return train_model

def _lazy_import_evaluate():
    from src.evaluate import evaluate_model, prepare_model_signature
    return evaluate_model, prepare_model_signature

__all__ = [
    "Classifier",
    "RiceDiseaseDataset",
    "config",
]