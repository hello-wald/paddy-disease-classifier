"""
MLflow utilities for rice disease classification project
This file contains helper functions for MLflow experiment management
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def start_mlflow_ui():
    """
    Instructions to start MLflow UI
    Run this command in terminal: mlflow ui
    Then open http://localhost:5000 in your browser
    """
    print("To start MLflow UI, run the following command in your terminal:")
    print("mlflow ui")
    print("Then open http://localhost:5000 in your browser")

def compare_experiments(experiment_name="Rice Disease Classification", top_n=5):
    """
    Compare top N runs from an experiment
    """
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.final_accuracy DESC"],
        max_results=top_n
    )
    
    if not runs:
        print("No runs found in the experiment")
        return
    
    # Create comparison dataframe
    comparison_data = []
    for run in runs:
        run_data = {
            'run_id': run.info.run_id,
            'start_time': run.info.start_time,
            'status': run.info.status
        }
        
        # Add metrics
        for key, value in run.data.metrics.items():
            run_data[key] = value
            
        # Add important parameters
        for key, value in run.data.params.items():
            if key in ['learning_rate', 'batch_size', 'num_epochs', 'model_architecture']:
                run_data[key] = value
                
        comparison_data.append(run_data)
    
    df = pd.DataFrame(comparison_data)
    print("Top runs comparison:")
    print(df.to_string(index=False))
    
    return df

def plot_experiment_metrics(experiment_name="Rice Disease Classification", metric_name="final_accuracy"):
    """
    Plot metrics across all runs in an experiment
    """
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    if not runs:
        print("No runs found in the experiment")
        return
    
    # Extract metrics
    run_metrics = []
    for run in runs:
        if metric_name in run.data.metrics:
            run_metrics.append({
                'run_id': run.info.run_id[:8],  # Short run ID
                'metric_value': run.data.metrics[metric_name],
                'learning_rate': run.data.params.get('learning_rate', 'unknown'),
                'batch_size': run.data.params.get('batch_size', 'unknown')
            })
    
    if not run_metrics:
        print(f"No runs found with metric '{metric_name}'")
        return
    
    df = pd.DataFrame(run_metrics)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(df)), df['metric_value'])
    plt.xlabel('Run')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} Across Runs')
    plt.xticks(range(len(df)), df['run_id'], rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def load_best_model(experiment_name="Rice Disease Classification"):
    """
    Load the best model from an experiment
    """
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    
    # Get best run by accuracy
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.final_accuracy DESC"],
        max_results=1
    )
    
    if not runs:
        print("No runs found in the experiment")
        return None
    
    best_run = runs[0]
    print(f"Loading best model from run: {best_run.info.run_id}")
    print(f"Best accuracy: {best_run.data.metrics.get('final_accuracy', 'N/A')}")
    
    # Load model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    return model, best_run

def log_model_info():
    """
    Example of logging additional model information
    """
    # This can be called during training to log additional info
    model_info = {
        "framework": "PyTorch",
        "architecture": "EfficientNet-B4",
        "pretrained": True,
        "task": "Multi-class Classification",
        "classes": ["Bacterial Blight", "Brown Spot", "Rice Blast"],
        "input_size": "224x224",
        "normalization": "ImageNet standard"
    }
    
    for key, value in model_info.items():
        mlflow.log_param(f"model_{key}", value)

def cleanup_old_runs(experiment_name="Rice Disease Classification", keep_top_n=10):
    """
    Delete old runs, keeping only the top N by accuracy
    """
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    # Get all runs sorted by accuracy
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.final_accuracy DESC"]
    )
    
    if len(runs) <= keep_top_n:
        print(f"Only {len(runs)} runs found, nothing to delete")
        return
    
    # Delete runs beyond top N
    runs_to_delete = runs[keep_top_n:]
    
    print(f"Deleting {len(runs_to_delete)} old runs, keeping top {keep_top_n}")
    
    for run in runs_to_delete:
        client.delete_run(run.info.run_id)
        print(f"Deleted run: {run.info.run_id}")

if __name__ == "__main__":
    # Example usage
    print("MLflow Utilities for Rice Disease Classification")
    print("=" * 50)
    
    # Show how to start UI
    start_mlflow_ui()
    print()
    
    # Compare experiments (will only work if you have run experiments)
    try:
        df = compare_experiments()
        if df is not None and len(df) > 0:
            plot_experiment_metrics()
    except Exception as e:
        print(f"No experiments to compare yet: {e}")
