import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import UrduClinicalEmotionTransformer
import wandb
import numpy as np
from tqdm import tqdm
from dataset import UrduEmotionDataset, collate_fn
from audio_preprocessing import AudioPreprocessor
from datetime import datetime
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Best hyperparameters from tuning
config = {
    'hidden_dim': 256,
    'num_layers': 6,
    'num_heads': 2,
    'ff_expansion': 4,
    'conv_kernel': 15,
    'dropout': 0.1,
    'batch_size': 16,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'val_split': 0.2,
    'data_path': 'raw_data'
}

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def plot_metrics(metrics, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Acc')
    plt.plot(metrics['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_pr_curves(all_labels, all_probs, save_dir):
    n_classes = 4
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            (all_labels == i).astype(int),
            all_probs[:, i]
        )
        avg_precision = average_precision_score(
            (all_labels == i).astype(int),
            all_probs[:, i]
        )
        
        plt.plot(recall, precision,
                label=f'Class {i} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'))
    plt.close()

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            probs = torch.softmax(output, dim=1)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(target.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return (
        total_loss / len(val_loader),
        100. * correct / total,
        all_probs,
        all_labels
    )

def best_model(custom_config=None):
    """
    Main training function that can be imported and run
    
    Args:
        custom_config (dict, optional): Override default config with custom values
    """
    # Update config with custom values if provided
    global config
    if custom_config:
        config.update(custom_config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize preprocessing and datasets
    print("\nInitializing preprocessing and datasets...")
    preprocessor = AudioPreprocessor()
    
    # Create full dataset
    full_dataset = UrduEmotionDataset(config['data_path'], preprocessor)
    print(f"Total dataset size: {len(full_dataset)} files")
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * config['val_split'])
    train_size = total_size - val_size
    
    print(f"Splitting dataset: {train_size} training, {val_size} validation")
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Initialize model with best hyperparameters
    model = UrduClinicalEmotionTransformer(
        num_emotions=4,  # Angry, Happy, Neutral, Sad
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_expansion=config['ff_expansion'],
        conv_kernel=config['conv_kernel'],
        dropout=config['dropout']
    ).to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Setup wandb
    wandb.init(
        project="urdu-clinical-emotion",
        config=config,
        name="best_model_training"
    )
    
    # Create output directory
    output_dir = 'experiments'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nStarting training with device: {device}")
    print(f"Model directory: {model_dir}")
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    # Initialize metrics storage
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch: {epoch+1}/{config["num_epochs"]}')
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        val_loss, val_acc, val_probs, val_labels = validate(model, val_loader, criterion, device)
        
        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch': epoch
        })
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Calculate F1 Score
        val_preds = np.argmax(val_probs, axis=1)
        f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f'Validation F1 Score: {f1:.4f}')
        wandb.log({'f1_score': f1})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(model_dir, 'best_model_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'f1_score': f1
            }, checkpoint_path)
            print(f'Saved new best model with val_loss: {val_loss:.4f}')
            
            # Plot and save PR curves for best model
            plot_pr_curves(val_labels, val_probs, model_dir)
        
        # Early stopping check
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
    
    # Plot final training curves
    plot_metrics(metrics, model_dir)
    print(f"\nTraining completed. Results saved in: {model_dir}")
    
    return model, metrics

if __name__ == '__main__':
    best_model()
