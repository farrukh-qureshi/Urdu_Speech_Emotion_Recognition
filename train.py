import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def train_model(model, train_loader, val_loader, num_epochs=50, debug=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Initialize optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    if debug:
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
            return
            
        except Exception as e:
            print(f"\nDebug check failed: {str(e)}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            return
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
        # Print training stats
        print(f'\nEpoch: {epoch}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {100.*train_correct/train_total:.2f}%')
        
        scheduler.step()
        
        # Save checkpoint every N epochs
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, f'checkpoints/model_epoch_{epoch}.pt')

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