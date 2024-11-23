import torch
import torch.nn as nn
from models import UrduClinicalEmotionTransformer
from train import train_model
from dataset import UrduEmotionDataset, collate_fn
from torch.utils.data import DataLoader
from audio_preprocessing import AudioPreprocessor
import itertools
import json
from datetime import datetime
import os
from visualization import PerformanceTracker

class ModelConfig:
    def __init__(self, **kwargs):
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.num_layers = kwargs.get('num_layers', 6)
        self.num_heads = kwargs.get('num_heads', 4)
        self.ff_expansion = kwargs.get('ff_expansion', 2)
        self.conv_kernel = kwargs.get('conv_kernel', 15)
        self.dropout = kwargs.get('dropout', 0.1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(config, train_loader, val_loader, experiment_dir):
    # Create tracker for this experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"exp_{timestamp}"
    tracker = PerformanceTracker(os.path.join(experiment_dir, exp_name))
    tracker.set_config(vars(config))
    
    # Initialize model
    model = UrduClinicalEmotionTransformer(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_expansion=config.ff_expansion,
        conv_kernel=config.conv_kernel,
        dropout=config.dropout
    )
    
    # Train model with tracker
    best_val_loss, best_val_acc = train_model(
        model, 
        train_loader, 
        val_loader,
        tracker=tracker,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_epochs=20,  # Reduced epochs for quick testing
        experiment_dir=experiment_dir
    )
    
    # Generate visualizations
    class_names = ['Angry', 'Happy', 'Neutral', 'Sad']
    tracker.plot_training_curves()
    tracker.plot_confusion_matrix(class_names)
    tracker.plot_learning_rate()
    tracker.plot_epoch_times()
    tracker.generate_classification_report(class_names)
    tracker.export_metrics()
    
    return {
        'params': count_parameters(model),
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'config': vars(config),
        'experiment_dir': os.path.join(experiment_dir, exp_name)
    }

def grid_search():
    # Define parameter grid
    param_grid = {
        'hidden_dim': [128, 256, 384],
        'num_layers': [4, 6, 8],
        'num_heads': [2, 4, 6],
        'ff_expansion': [2, 4],
        'conv_kernel': [7, 15],
        'dropout': [0.1, 0.2],
        'batch_size': [16, 32],
        'learning_rate': [1e-3, 1e-4]
    }
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f'experiments/grid_search_{timestamp}'
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save parameter grid
    with open(f'{experiment_dir}/param_grid.json', 'w') as f:
        json.dump(param_grid, f, indent=4)
    
    # Initialize dataset and loaders
    preprocessor = AudioPreprocessor()
    
    # Load the full dataset
    full_dataset = UrduEmotionDataset('raw_data', preprocessor)
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)  # 20% for validation
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Store results
    results = []
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    for params in itertools.product(*values):
        config = ModelConfig(**dict(zip(keys, params)))
        
        # Create data loaders with current batch size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            collate_fn=collate_fn
        )
        
        print("\n" + "="*50)
        print("Running experiment with config:")
        for k, v in vars(config).items():
            print(f"{k}: {v}")
        
        try:
            result = run_experiment(config, train_loader, val_loader, experiment_dir)
            results.append(result)
            
            # Save intermediate results
            with open(f'{experiment_dir}/results.json', 'w') as f:
                json.dump(results, f, indent=4)
                
            print("\nResults:")
            print(f"Parameters: {result['params']:,}")
            print(f"Val Loss: {result['val_loss']:.4f}")
            print(f"Val Accuracy: {result['val_acc']:.2f}%")
            
        except Exception as e:
            print(f"\nExperiment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Analyze results
    analyze_results(results, experiment_dir)

def analyze_results(results, experiment_dir):
    # Sort by different metrics
    by_params = sorted(results, key=lambda x: x['params'])
    by_accuracy = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    by_efficiency = sorted(results, key=lambda x: x['val_acc'] / x['params'], reverse=True)
    
    analysis = {
        'smallest_model': by_params[0],
        'largest_model': by_params[-1],
        'best_accuracy': by_accuracy[0],
        'best_efficiency': by_efficiency[0],
        'top_5_accuracy': by_accuracy[:5],
        'top_5_efficiency': by_efficiency[:5]
    }
    
    # Save analysis
    with open(f'{experiment_dir}/analysis.json', 'w') as f:
        json.dump(analysis, f, indent=4)
    
    # Print summary
    print("\n" + "="*50)
    print("Grid Search Results Summary")
    print("="*50)
    print("\nBest Accuracy Model:")
    print_config(analysis['best_accuracy'])
    
    print("\nMost Efficient Model (Accuracy/Parameters):")
    print_config(analysis['best_efficiency'])

def print_config(result):
    print(f"Parameters: {result['params']:,}")
    print(f"Validation Accuracy: {result['val_acc']:.2f}%")
    print(f"Validation Loss: {result['val_loss']:.4f}")
    print("\nConfiguration:")
    for k, v in result['config'].items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    grid_search() 