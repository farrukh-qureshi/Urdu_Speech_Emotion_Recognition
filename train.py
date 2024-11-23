import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn
import os
import time

def save_checkpoint(model, optimizer, epoch, loss, path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def train_model(model, train_loader, val_loader, tracker=None, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Initialize optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    if kwargs.get('debug', True):
        print("\nRunning debug checks...")
        try:
            # Enable gradients
            torch.set_grad_enabled(True)
            
            # Get a single batch
            data, target = next(iter(train_loader))
            print(f"Batch shapes - Input: {data.shape}, Target: {target.shape}")
            
            # Move to device
            data, target = data.to(device), target.to(device)
            
            # Test forward pass
            print("Testing forward pass...")
            output = model(data)
            print(f"Output shape: {output.shape}")
            print(f"Output device: {output.device}")
            
            # Test loss computation
            print("\nTesting loss computation...")
            loss = criterion(output, target)
            print(f"Loss value: {loss.item()}")
            
            # Test backward pass
            print("\nTesting backward pass...")
            optimizer.zero_grad()
            loss.backward()
            print("Backward pass successful!")
            
            # Test optimizer step
            print("\nTesting optimizer step...")
            optimizer.step()
            optimizer.zero_grad()
            print("Optimizer step successful!")
            
            print("\nAll debug checks passed successfully!")
            
        except Exception as e:
            print(f"\nDebug check failed: {str(e)}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            return
    
    print("\nStarting actual training...")
    
    # Initialize optimizer and scheduler for actual training
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=kwargs.get('num_epochs', 50))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(kwargs.get('num_epochs', 50)):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{kwargs.get("num_epochs", 50)}')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        print("\nRunning validation...")
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{kwargs.get("num_epochs", 50)}')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{kwargs.get("num_epochs", 50)} Summary:')
        print(f'Training    - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation  - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                'checkpoints/best_model.pt'
            )
            print(f'New best model saved! (Val Loss: {val_loss:.4f})')
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f'checkpoints/model_epoch_{epoch+1}.pt'
            )
            print(f'Checkpoint saved for epoch {epoch+1}')
        
        scheduler.step()
        print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 80)
        
        epoch_time = time.time() - epoch_start_time
        
        if tracker:
            tracker.update({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': scheduler.get_last_lr()[0],
                'epoch_time': epoch_time
            })
            tracker.update_predictions(val_labels, val_preds)

def main():
    # Initialize preprocessing and dataset
    preprocessor = AudioPreprocessor()
    train_dataset = UrduEmotionDataset('raw_data/train', preprocessor)
    val_dataset = UrduEmotionDataset('raw_data/val', preprocessor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = UrduClinicalEmotionTransformer()
    
    # Train model
    train_model(model, train_loader, val_loader)

if __name__ == '__main__':
    main() 