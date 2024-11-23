import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Tuple

class AudioPreprocessor:
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 win_length: int = 400,
                 hop_length: int = 160,
                 n_fft: int = 512):
        self.target_sr = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # SpecAugment parameters
        self.time_masking = T.TimeMasking(time_mask_param=40)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=20)
    
    def preprocess(self, audio_input, augment: bool = True) -> torch.Tensor:
        """
        Process either an audio file path or a waveform
        """
        if isinstance(audio_input, str):
            # Load and preprocess audio if path is provided
            waveform, sr = torchaudio.load(audio_input)
            if sr != self.target_sr:
                resampler = T.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            # Use the provided waveform directly
            waveform = audio_input
        
        # Extract mel spectrogram
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db  # [1, n_mels, time]