import json
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import os
from typing import Dict, List, Any

class EnhancedEmotionAnalyzer:
    def __init__(self, json_path: str, audio_dir: str, output_dir: str = 'enhanced_statistical_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = Path(audio_dir)
        
        # Load the JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.emotions = list(self.data['dataset_summary']['emotion_distribution'].keys())
        
    def extract_enhanced_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Extract enhanced set of features from audio file"""
        y, sr = librosa.load(audio_path)
        
        # Basic features
        zero_crossings = librosa.zero_crossings(y).sum()
        rms = librosa.feature.rms(y=y).mean()
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        
        # Mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = mel_spec_db.mean()
        mel_std = mel_spec_db.std()
        mel_skew = stats.skew(mel_spec_db.flatten())
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1)
        mfcc_stds = mfccs.std(axis=1)
        
        return {
            'basic_features': {
                'zero_crossings': zero_crossings,
                'rms_energy': rms,
                'spectral_centroid': spec_cent,
                'spectral_rolloff': spec_rolloff
            },
            'mel_features': {
                'mel_mean': mel_mean,
                'mel_std': mel_std,
                'mel_skewness': mel_skew
            },
            'mfcc_features': {
                'mfcc_means': mfcc_means,
                'mfcc_stds': mfcc_stds
            }
        }
    
    def collect_emotion_features(self) -> Dict[str, Dict[str, List[float]]]:
        """Collect features for all audio files grouped by emotion"""
        emotion_features = {emotion: {
            'basic_features': {},
            'mel_features': {},
            'mfcc_features': {}
        } for emotion in self.emotions}
        
        for emotion in self.emotions:
            emotion_dir = self.audio_dir / emotion
            if not emotion_dir.exists():
                continue
                
            audio_files = list(emotion_dir.glob('*.wav'))
            features_list = []
            
            for audio_file in audio_files:
                features = self.extract_enhanced_features(str(audio_file))
                features_list.append(features)
            
            # Aggregate features
            for feature_type in ['basic_features', 'mel_features', 'mfcc_features']:
                for feature_name in features_list[0][feature_type]:
                    values = [f[feature_type][feature_name] for f in features_list]
                    if isinstance(values[0], np.ndarray):
                        values = [v.tolist() for v in values]
                    emotion_features[emotion][feature_type][feature_name] = values
        
        return emotion_features
    
    def perform_statistical_tests(self, emotion_features: Dict) -> Dict:
        """Perform statistical tests on all features"""
        results = {
            'anova_results': {},
            'effect_sizes': {},
            'tukey_results': {}
        }
        
        # Analyze each feature type
        for feature_type in ['basic_features', 'mel_features', 'mfcc_features']:
            for feature_name in emotion_features[self.emotions[0]][feature_type]:
                # Prepare data for analysis
                feature_data = []
                for emotion in self.emotions:
                    values = emotion_features[emotion][feature_type][feature_name]
                    if isinstance(values[0], list):
                        values = [v[0] for v in values]  # Take first coefficient for MFCC
                    feature_data.append(values)
                
                # ANOVA test
                f_stat, p_value = stats.f_oneway(*feature_data)
                results['anova_results'][f"{feature_type}_{feature_name}"] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value)
                }
                
                # Effect sizes
                effect_size_matrix = np.zeros((len(self.emotions), len(self.emotions)))
                for i in range(len(self.emotions)):
                    for j in range(i+1, len(self.emotions)):
                        d = self.cohens_d(feature_data[i], feature_data[j])
                        effect_size_matrix[i, j] = d
                        effect_size_matrix[j, i] = -d
                
                results['effect_sizes'][f"{feature_type}_{feature_name}"] = {
                    'matrix': effect_size_matrix.tolist(),
                    'emotions': self.emotions
                }
                
                # Tukey's test
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                flat_data = np.concatenate(feature_data)
                flat_labels = np.concatenate([[em] * len(fd) for em, fd in zip(self.emotions, feature_data)])
                tukey = pairwise_tukeyhsd(flat_data, flat_labels)
                results['tukey_results'][f"{feature_type}_{feature_name}"] = str(tukey)
        
        return results
    
    @staticmethod
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se
    
    def visualize_results(self, results: Dict):
        """Create visualizations for the statistical analysis results"""
        # Effect size heatmaps
        for feature_name, effect_size_data in results['effect_sizes'].items():
            matrix = np.array(effect_size_data['matrix'])
            emotions = effect_size_data['emotions']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, 
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       xticklabels=emotions, 
                       yticklabels=emotions)
            plt.title(f'Effect Sizes: {feature_name}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'effect_sizes_{feature_name}.png')
            plt.close()
        
        # Save numerical results
        with open(self.output_dir / 'enhanced_statistical_analysis.json', 'w') as f:
            json.dump(results, f, indent=4)

def main():
    analyzer = EnhancedEmotionAnalyzer(
        'EDA/dataset_statistics.json',
        'raw_data'
    )
    
    # Collect features
    print("Collecting features...")
    emotion_features = analyzer.collect_emotion_features()
    
    # Perform analysis
    print("Performing statistical analysis...")
    results = analyzer.perform_statistical_tests(emotion_features)
    
    # Visualize results
    print("Creating visualizations...")
    analyzer.visualize_results(results)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("\nANOVA Results:")
    for feature, result in results['anova_results'].items():
        if result['p_value'] < 0.05:
            print(f"\n{feature}:")
            print(f"F-statistic: {result['f_statistic']:.4f}")
            print(f"p-value: {result['p_value']:.4f}")

if __name__ == "__main__":
    main() 