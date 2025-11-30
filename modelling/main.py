import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import torch
from torch.utils.data import random_split, Subset
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Import modules
from models.classifier import Classifier
from data.dataset import RiceDiseaseDataset
from utils.transforms import get_transforms
from utils.losses import FocalLoss
from utils.visualization import plot_training_history, plot_confusion_matrix
from training.trainer import train_model
from training.evaluator import evaluate_model, prepare_model_signature
import config

# Make sure the graphs directory exists
os.makedirs(config.GRAPH_DIR, exist_ok=True)

def main():
    """Main training pipeline with MLflow tracking"""
    
    mlflow.set_experiment("Rice Disease Classification")
    
    with mlflow.start_run():
        mlflow.log_param("batch_size", config.BATCH_SIZE)
        mlflow.log_param("device", str(config.DEVICE))
        mlflow.log_param("model_name", config.MODEL_NAME)
        
        if not os.path.exists(config.TRAIN_DATA_DIR):
            print(f"Error: Training directory {config.TRAIN_DATA_DIR} does not exist!")
            return
        if not os.path.exists(config.TEST_DATA_DIR):
            print(f"Error: Test directory {config.TEST_DATA_DIR} does not exist!")
            return

        train_transform, val_transform = get_transforms()

        print("\nCreating training dataset...")
        train_dataset = RiceDiseaseDataset(config.TRAIN_DATA_DIR, transform=train_transform)
        print("\nCreating test dataset...")
        test_dataset = RiceDiseaseDataset(config.TEST_DATA_DIR, transform=val_transform)

        train_size = int(0.8 * len(train_dataset)) 
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        val_indices = val_dataset.indices
        print("\nCreating validation dataset...")
        val_dataset = RiceDiseaseDataset(config.TRAIN_DATA_DIR, transform=val_transform)
        val_dataset = Subset(val_dataset, val_indices)

        if len(train_dataset) == 0:
            print("Error: No training images found!")
            return
        if len(val_dataset) == 0:
            print("Error: No validation images found!")
            return
        if len(test_dataset) == 0:
            print("Error: No test images found!")
            return

        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("val_samples", len(val_dataset))
        mlflow.log_param("test_samples", len(test_dataset))
        mlflow.log_param("num_classes", config.NUM_CLASSES)
        mlflow.log_param("class_names", config.CLASS_NAMES)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=config.NUM_WORKERS, 
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=config.NUM_WORKERS,
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=config.NUM_WORKERS,
        )

        print(f"\nTraining samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        model = Classifier(num_classes=config.NUM_CLASSES, model_name=config.MODEL_NAME, classifier_units=config.CLASSIFIER_UNITS, activation_function=config.ACTIVATION_FUNCTION)
        model = model.to(config.DEVICE)
        
        criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")
        
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)
        mlflow.log_param("focal_loss_alpha", config.FOCAL_LOSS_ALPHA)
        mlflow.log_param("focal_loss_gamma", config.FOCAL_LOSS_GAMMA)
        mlflow.log_param("classifier_units", config.CLASSIFIER_UNITS)
        mlflow.log_param("activation_function", config.ACTIVATION_FUNCTION)
        
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, 
            config.NUM_EPOCHS, config.LEARNING_RATE, config.DEVICE
        )
        
        plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        model = model.to(config.DEVICE)
        
        accuracy, report, cm = evaluate_model(model, test_loader, config.CLASS_NAMES, config.DEVICE)
        plot_confusion_matrix(cm, config.CLASS_NAMES)
        
        signature, input_example = prepare_model_signature(model, test_loader, config.DEVICE)
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="rice_disease_classifier",
            signature=signature,
            input_example=input_example  
        )
        
        mlflow.log_artifact(config.MODEL_SAVE_PATH)
        
        print("Training completed!")
        print(f"Best model saved as '{config.MODEL_SAVE_PATH}'")
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow experiment ID: {mlflow.active_run().info.experiment_id}")

def predict_single_image(image_path):
    """Predict disease for a single image"""
    
    model = Classifier(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    model = model.to(config.DEVICE)
    
    _, val_transform = get_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return config.CLASS_NAMES[predicted_class], confidence

if __name__ == "__main__":
    main()