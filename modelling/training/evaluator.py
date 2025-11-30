import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
from mlflow.models.signature import infer_signature

def evaluate_model(model, test_loader, class_names, device): 
    """Evaluate the model with the test dataset"""

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    mlflow.log_metric("final_accuracy", accuracy)
    
    for class_name in class_names:
        if class_name in report:
            mlflow.log_metric(f"{class_name}_precision", report[class_name]['precision'])
            mlflow.log_metric(f"{class_name}_recall", report[class_name]['recall'])
            mlflow.log_metric(f"{class_name}_f1", report[class_name]['f1-score'])
    
    mlflow.log_metric("macro_avg_precision", report['macro avg']['precision'])
    mlflow.log_metric("macro_avg_recall", report['macro avg']['recall'])
    mlflow.log_metric("macro_avg_f1", report['macro avg']['f1-score'])
    mlflow.log_metric("weighted_avg_precision", report['weighted avg']['precision'])
    mlflow.log_metric("weighted_avg_recall", report['weighted avg']['recall'])
    mlflow.log_metric("weighted_avg_f1", report['weighted avg']['f1-score'])
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return accuracy, report, cm

def prepare_model_signature(model, test_loader, device):
    """Prepare model signature for MLflow"""
    
    for sample_input, _ in test_loader:
        break
    
    sample_input = sample_input[0:1].to(device)
    
    with torch.no_grad():
        sample_output = model(sample_input)
    
    input_np = sample_input.cpu().numpy()
    output_np = sample_output.cpu().numpy()
    
    signature = infer_signature(input_np, output_np)
    
    return signature, input_np