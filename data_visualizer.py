import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from pathlib import Path
from audio_preprocessing import AudioPreprocessor
from dataset import UrduEmotionDataset
import torchaudio
from typing import Dict, Any

class DataPipelineVisualizer:
    def __init__(self, output_dir: str = 'visualization_output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = AudioPreprocessor()
        
    def visualize_single_sample(self, audio_path: str) -> Dict[str, Any]:
        """Create visualizations for a single audio sample through the entire pipeline"""
        
        # Create sample-specific directory
        sample_name = Path(audio_path).stem
        sample_dir = self.output_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        # Dictionary to store all the processing information
        processing_info = {
            "file_name": sample_name,
            "processing_steps": {}
        }
        
        # 1. Raw Audio Waveform
        y, sr = librosa.load(audio_path)
        processing_info["processing_steps"]["raw_audio"] = {
            "sampling_rate": sr,
            "duration": len(y) / sr,
            "shape": y.shape
        }
        
        plt.figure(figsize=(12, 4))
        plt.plot(y)
        plt.title("Raw Audio Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.savefig(sample_dir / "1_raw_waveform.png")
        plt.close()
        
        # 2. Mel Spectrogram (before normalization)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        processing_info["processing_steps"]["mel_spectrogram"] = {
            "shape": mel_spec.shape,
            "min_value": float(mel_spec_db.min()),
            "max_value": float(mel_spec_db.max())
        }
        
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram (before normalization)")
        plt.savefig(sample_dir / "2_mel_spectrogram.png")
        plt.close()
        
        # 3. Processed Features (after preprocessor)
        processed_features = self.preprocessor.preprocess(audio_path, augment=False)
        processing_info["processing_steps"]["processed_features"] = {
            "shape": list(processed_features.shape),
            "min_value": float(processed_features.min()),
            "max_value": float(processed_features.max())
        }
        
        plt.figure(figsize=(12, 4))
        plt.imshow(processed_features.squeeze().numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.title("Processed Features (Model Input)")
        plt.xlabel("Time")
        plt.ylabel("Mel Bins")
        plt.savefig(sample_dir / "3_processed_features.png")
        plt.close()
        
        # 4. Feature Distribution
        plt.figure(figsize=(10, 4))
        sns.histplot(processed_features.numpy().flatten(), bins=50)
        plt.title("Distribution of Feature Values")
        plt.xlabel("Feature Value")
        plt.ylabel("Count")
        plt.savefig(sample_dir / "4_feature_distribution.png")
        plt.close()
        
        # Save processing information
        with open(sample_dir / "processing_info.json", 'w') as f:
            json.dump(processing_info, f, indent=4)
        
        return processing_info
    
    def visualize_batch(self, dataset: UrduEmotionDataset, batch_size: int = 4):
        """Visualize a batch of samples"""
        batch_dir = self.output_dir / "batch_visualization"
        batch_dir.mkdir(exist_ok=True)
        
        # Get a batch of samples
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        batch_samples = [dataset[i] for i in indices]
        features = [sample[0] for sample in batch_samples]
        labels = [sample[1] for sample in batch_samples]
        
        # Visualize batch statistics
        batch_info = {
            "batch_size": batch_size,
            "feature_shapes": [list(f.shape) for f in features],
            "labels": labels
        }
        
        # Plot batch features
        plt.figure(figsize=(15, 10))
        for idx, feature in enumerate(features):
            plt.subplot(batch_size, 1, idx + 1)
            plt.imshow(feature.squeeze().numpy(), aspect='auto', origin='lower')
            plt.title(f"Sample {idx} (Label: {labels[idx]})")
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(batch_dir / "batch_features.png")
        plt.close()
        
        # Save batch information
        with open(batch_dir / "batch_info.json", 'w') as f:
            json.dump(batch_info, f, indent=4)

def main():
    # Initialize visualizer
    visualizer = DataPipelineVisualizer()
    
    # Example usage
    data_path = "raw_data"  # Update with your data path
    preprocessor = AudioPreprocessor()
    dataset = UrduEmotionDataset(data_path, preprocessor)
    
    # Visualize a few random samples
    for emotion in ['Angry', 'Happy', 'Neutral', 'Sad']:
        emotion_path = os.path.join(data_path, emotion)
        if os.path.exists(emotion_path):
            audio_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            if audio_files:
                sample_file = os.path.join(emotion_path, audio_files[0])
                print(f"Processing {emotion} sample: {sample_file}")
                visualizer.visualize_single_sample(sample_file)
    
    # Visualize a batch
    print("Visualizing batch...")
    visualizer.visualize_batch(dataset)
    
    print(f"Visualizations saved to {visualizer.output_dir}")

if __name__ == "__main__":
    main() 