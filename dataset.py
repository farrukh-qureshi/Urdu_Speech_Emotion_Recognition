import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, List

class UrduEmotionDataset(Dataset):
    def __init__(self, data_path: str, preprocessor, augment: bool = True):
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.augment = augment
        self.samples = self._load_dataset()
        self.emotion_to_idx = {
            'Angry': 0,
            'Happy': 1,
            'Neutral': 2,
            'Sad': 3
        }
    
    def _load_dataset(self) -> List[Tuple[str, str]]:
        samples = []
        for emotion in ['Angry', 'Happy', 'Neutral', 'Sad']:
            emotion_path = os.path.join(self.data_path, emotion)
            if not os.path.exists(emotion_path):
                continue
            
            for file in os.listdir(emotion_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(emotion_path, file)
                    samples.append((file_path, emotion))
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, emotion = self.samples[idx]
        features = self.preprocessor.preprocess(file_path, self.augment)
        emotion_idx = self.emotion_to_idx[emotion]
        return features, emotion_idx 

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Separate features and labels
    features = [item[0] for item in batch]  # Each feature is [1, n_mels, time]
    labels = [item[1] for item in batch]
    
    # Print shapes for debugging
    # print(f"Feature shapes: {[f.shape for f in features]}")
    
    # Get max length in the batch
    max_len = max([f.size(-1) for f in features])
    
    # Pad features to max length
    padded_features = []
    for feat in features:
        # Calculate padding
        pad_len = max_len - feat.size(-1)
        if pad_len > 0:
            padded_feat = F.pad(feat, (0, pad_len))
        else:
            padded_feat = feat
        padded_features.append(padded_feat)
    
    # Stack features and labels
    features_tensor = torch.stack(padded_features)  # [batch_size, 1, n_mels, time]
    labels_tensor = torch.tensor(labels)
    
    return features_tensor, labels_tensor