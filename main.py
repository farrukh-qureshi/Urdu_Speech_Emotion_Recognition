import os
import torch
from stats_analyzer import DatasetAnalyzer
from audio_preprocessing import AudioPreprocessor
from models import UrduClinicalEmotionTransformer
from train import train_model
from torch.utils.data import DataLoader, random_split
from dataset import UrduEmotionDataset, collate_fn

def main():
    # Configuration
    data_path = 'raw_data'
    batch_size = 2  # Reduced for testing
    num_epochs = 1
    val_split = 0.2  # 20% for validation
    
    print("\n=== Starting Dataset Analysis ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Looking for data in: {os.path.abspath(data_path)}")
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print(f"Error: {data_path} directory not found")
        return
    
    # Verify emotion directories exist
    expected_emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
    for emotion in expected_emotions:
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"Error: {emotion} directory not found in {data_path}")
            return
        else:
            print(f"Found {emotion} directory with {len(os.listdir(emotion_path))} files")
    
    # 1. Analyze Dataset
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
    
    # 2. Initialize preprocessing and datasets
    print("\nInitializing preprocessing and datasets...")
    preprocessor = AudioPreprocessor()
    
    # Create full dataset
    full_dataset = UrduEmotionDataset(data_path, preprocessor)
    print(f"Total dataset size: {len(full_dataset)} files")
    
    # 3. Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    print(f"Splitting dataset: {train_size} training, {val_size} validation")
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 4. Create data loaders
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
    
    # 5. Initialize model
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
    
    # 6. Train model
    print("\nStarting training...")
    try:
        # Set debug=True to run checks without full training
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, debug=True)
        print("Debug checks completed successfully!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
