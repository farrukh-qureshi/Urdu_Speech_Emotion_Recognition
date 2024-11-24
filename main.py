import os
import torch
from stats_analyzer import DatasetAnalyzer
from audio_preprocessing import AudioPreprocessor
from models import UrduClinicalEmotionTransformer
from train import train_model
from torch.utils.data import DataLoader, random_split
from dataset import UrduEmotionDataset, collate_fn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class ExperimentTracker:
    def __init__(self, output_dir='experiments'):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, f'run_{self.timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'learning_rate': []
        }
        
        # Store predictions for confusion matrix
        self.val_predictions = []
        self.val_targets = []

    def update(self, metrics_dict):
        """Update metrics from a dictionary of values"""
        epoch = len(self.metrics['epoch'])
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(metrics_dict['train_loss'])
        self.metrics['train_acc'].append(metrics_dict['train_acc'])
        self.metrics['val_loss'].append(metrics_dict['val_loss'])
        self.metrics['val_acc'].append(metrics_dict['val_acc'])
        self.metrics['learning_rate'].append(metrics_dict['lr'])
        
        # Save after each update
        self.save_metrics()

    def update_predictions(self, targets, predictions):
        self.val_targets.extend(targets)
        self.val_predictions.extend(predictions)

    def save_metrics(self):
        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.data_dir, 'training_metrics.csv'), index=False)

    def plot_training_curves(self):
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['epoch'], self.metrics['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['epoch'], self.metrics['train_acc'], label='Train Acc')
        plt.plot(self.metrics['epoch'], self.metrics['val_acc'], label='Val Acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'))
        plt.close()

    def plot_confusion_matrix(self, class_names):
        cm = confusion_matrix(self.val_targets, self.val_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'))
        plt.close()

    def save_classification_report(self, class_names):
        report = classification_report(
            self.val_targets, 
            self.val_predictions,
            target_names=class_names,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.data_dir, 'classification_report.csv'))

def main():
    # Configuration
    data_path = 'raw_data'
    batch_size = 16
    num_epochs = 100
    val_split = 0.2
    expected_emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    print("\n=== Starting Dataset Analysis ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Looking for data in: {os.path.abspath(data_path)}")
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print(f"Error: {data_path} directory not found")
        return
    
    # Verify emotion directories exist
    for emotion in expected_emotions:
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"Error: {emotion} directory not found in {data_path}")
            return
        else:
            print(f"Found {emotion} directory with {len(os.listdir(emotion_path))} files")
    # # 1. Analyze Dataset
    # print("\nInitializing DatasetAnalyzer...")
    # analyzer = DatasetAnalyzer(data_path)
    
    # print("Starting dataset analysis...")
    # try:
    #     stats = analyzer.analyze_dataset()
    #     print("Analysis completed. Generating summary...")
    #     analyzer.print_stats_summary(stats)
    #     print("\nCreating visualization plots...")
    #     analyzer.plot_statistics(stats)
    #     print("Analysis and visualization completed successfully!")
    # except Exception as e:
    #     print(f"\nError during analysis: {str(e)}")
    #     import traceback
    #     print("\nFull traceback:")
    #     traceback.print_exc()
    #     return

    # print("\nDataset analysis completed!")

    # Initialize preprocessing and datasets
    print("\nInitializing preprocessing and datasets...")
    preprocessor = AudioPreprocessor()
    
    # Create full dataset
    full_dataset = UrduEmotionDataset(data_path, preprocessor)
    print(f"Total dataset size: {len(full_dataset)} files")
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    print(f"Splitting dataset: {train_size} training, {val_size} validation")
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Initialize model with optimized configuration
    print("Initializing model...")
    model = UrduClinicalEmotionTransformer(num_emotions=len(expected_emotions))
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    try:
        train_model(model, train_loader, val_loader, 
                   num_epochs=num_epochs, debug=True, 
                   tracker=tracker)
        
        # Generate final plots and reports
        tracker.plot_training_curves()
        tracker.plot_confusion_matrix(expected_emotions)
        tracker.save_classification_report(expected_emotions)
        
        print("Training completed successfully!")
        print(f"\nResults saved in: {tracker.output_dir}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
