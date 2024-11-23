import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime

class PerformanceTracker:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.plots_dir = os.path.join(experiment_dir, 'plots')
        self.data_dir = os.path.join(experiment_dir, 'data')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize tracking dictionaries
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
        self.epoch_times = []
        self.predictions = []
        self.true_labels = []
        self.config = None
        
    def update(self, epoch_metrics):
        """Update metrics after each epoch"""
        self.train_losses.append(epoch_metrics['train_loss'])
        self.train_accs.append(epoch_metrics['train_acc'])
        self.val_losses.append(epoch_metrics['val_loss'])
        self.val_accs.append(epoch_metrics['val_acc'])
        self.learning_rates.append(epoch_metrics['lr'])
        self.epoch_times.append(epoch_metrics['epoch_time'])
        
    def update_predictions(self, y_true, y_pred):
        """Update predictions for confusion matrix"""
        self.predictions.extend(y_pred)
        self.true_labels.extend(y_true)
    
    def set_config(self, config):
        """Store model configuration"""
        self.config = config
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs, self.val_accs, 'r-', label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'))
        plt.close()
    
    def plot_confusion_matrix(self, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.true_labels, self.predictions)
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
    
    def plot_learning_rate(self):
        """Plot learning rate schedule"""
        epochs = range(1, len(self.learning_rates) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.learning_rates, 'g-')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_rate.png'))
        plt.close()
    
    def plot_epoch_times(self):
        """Plot epoch execution times"""
        epochs = range(1, len(self.epoch_times) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.epoch_times, 'm-')
        plt.title('Epoch Execution Times')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'epoch_times.png'))
        plt.close()
    
    def export_metrics(self):
        """Export all metrics to CSV"""
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'train_accuracy': self.train_accs,
            'val_loss': self.val_losses,
            'val_accuracy': self.val_accs,
            'learning_rate': self.learning_rates,
            'epoch_time': self.epoch_times
        })
        
        metrics_df.to_csv(os.path.join(self.data_dir, 'training_metrics.csv'), index=False)
        
        # Export configuration
        if self.config:
            config_df = pd.DataFrame([self.config])
            config_df.to_csv(os.path.join(self.data_dir, 'model_config.csv'), index=False)
    
    def generate_classification_report(self, class_names):
        """Generate and save classification report"""
        report = classification_report(self.true_labels, self.predictions, 
                                    target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.data_dir, 'classification_report.csv')) 