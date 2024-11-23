import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
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
            
        # Print training stats
        print(f'\nEpoch: {epoch}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {100.*train_correct/train_total:.2f}%')
        
        scheduler.step()

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