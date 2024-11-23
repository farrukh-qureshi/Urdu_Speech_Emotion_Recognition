import os
import librosa
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import json

class DatasetAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.eda_dir = 'EDA'
        os.makedirs(self.eda_dir, exist_ok=True)
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the dataset and return comprehensive statistics"""
        stats = {
            'total_files': 0,
            'by_emotion': defaultdict(int),
            'by_duration': {
                'min': float('inf'),
                'max': 0,
                'mean': 0,
                'std': 0,
                'durations': []
            },
            'by_gender': defaultdict(int),
            'by_speaker': defaultdict(lambda: {
                'count': 0,
                'emotions': defaultdict(int),
                'gender': None,
                'avg_duration': 0,
                'total_duration': 0
            }),
            'audio_properties': {
                'sample_rates': defaultdict(int),
                'n_channels': defaultdict(int),
                'bit_depths': defaultdict(int),
                'file_sizes': [],
                'zero_crossings': [],
                'rms_energy': [],
                'spectral_centroids': [],
                'spectral_rolloffs': [],
                'mfccs': []
            },
            'emotion_durations': defaultdict(list),
            'gender_durations': defaultdict(list)
        }
        
        print("Analyzing audio files...")
        for emotion in ['Angry', 'Happy', 'Neutral', 'Sad']:
            emotion_path = os.path.join(self.data_path, emotion)
            if not os.path.exists(emotion_path):
                continue
                
            for file in tqdm(os.listdir(emotion_path), desc=f"Processing {emotion}"):
                if not file.endswith('.wav'):
                    continue
                
                file_path = os.path.join(emotion_path, file)
                stats['total_files'] += 1
                stats['by_emotion'][emotion] += 1
                
                try:
                    # Basic audio properties
                    y, sr = librosa.load(file_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    # Get file info
                    file_info = sf.info(file_path)
                    stats['audio_properties']['sample_rates'][sr] += 1
                    stats['audio_properties']['n_channels'][file_info.channels] += 1
                    stats['audio_properties']['bit_depths'][file_info.subtype] += 1
                    stats['audio_properties']['file_sizes'].append(os.path.getsize(file_path) / 1024)  # KB
                    
                    # Audio features
                    zero_crossings = librosa.zero_crossings(y).sum()
                    rms = librosa.feature.rms(y=y).mean()
                    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
                    
                    stats['audio_properties']['zero_crossings'].append(zero_crossings)
                    stats['audio_properties']['rms_energy'].append(rms)
                    stats['audio_properties']['spectral_centroids'].append(spectral_centroids)
                    stats['audio_properties']['spectral_rolloffs'].append(spectral_rolloff)
                    stats['audio_properties']['mfccs'].append(mfccs)
                    
                    # Duration statistics
                    stats['by_duration']['durations'].append(duration)
                    stats['by_duration']['min'] = min(stats['by_duration']['min'], duration)
                    stats['by_duration']['max'] = max(stats['by_duration']['max'], duration)
                    stats['emotion_durations'][emotion].append(duration)
                    
                    # Speaker and gender info
                    speaker_info = file.split('_')[0]
                    gender = 'Male' if speaker_info[1] == 'M' else 'Female'
                    speaker_id = speaker_info[2:]
                    
                    stats['by_gender'][gender] += 1
                    stats['gender_durations'][gender].append(duration)
                    stats['by_speaker'][speaker_id]['count'] += 1
                    stats['by_speaker'][speaker_id]['emotions'][emotion] += 1
                    stats['by_speaker'][speaker_id]['gender'] = gender
                    stats['by_speaker'][speaker_id]['total_duration'] += duration
                    
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
        
        # Calculate aggregate statistics
        if stats['by_duration']['durations']:
            durations = np.array(stats['by_duration']['durations'])
            stats['by_duration']['mean'] = np.mean(durations)
            stats['by_duration']['std'] = np.std(durations)
            stats['by_duration']['quartiles'] = np.percentile(durations, [25, 50, 75])
        
        # Calculate average duration per speaker
        for speaker_id in stats['by_speaker']:
            if stats['by_speaker'][speaker_id]['count'] > 0:
                stats['by_speaker'][speaker_id]['avg_duration'] = (
                    stats['by_speaker'][speaker_id]['total_duration'] / 
                    stats['by_speaker'][speaker_id]['count']
                )
        
        return stats
    
    def print_stats_summary(self, stats: Dict[str, Any]) -> None:
        """Print a comprehensive summary of the dataset statistics and save to JSON"""
        # Create a formatted stats dictionary for JSON
        json_stats = {
            "dataset_summary": {
                "total_files": stats['total_files'],
                "emotion_distribution": {
                    emotion: {
                        "count": count,
                        "percentage": (count / stats['total_files']) * 100,
                        "avg_duration": np.mean(stats['emotion_durations'][emotion])
                    }
                    for emotion, count in stats['by_emotion'].items()
                },
                "gender_distribution": {
                    gender: {
                        "count": count,
                        "percentage": (count / stats['total_files']) * 100,
                        "avg_duration": np.mean(stats['gender_durations'][gender])
                    }
                    for gender, count in stats['by_gender'].items()
                },
                "duration_statistics": {
                    "min": float(stats['by_duration']['min']),
                    "max": float(stats['by_duration']['max']),
                    "mean": float(stats['by_duration']['mean']),
                    "std": float(stats['by_duration']['std']),
                    "quartiles": {
                        "25%": float(stats['by_duration']['quartiles'][0]),
                        "50%": float(stats['by_duration']['quartiles'][1]),
                        "75%": float(stats['by_duration']['quartiles'][2])
                    }
                },
                "audio_properties": {
                    "sample_rates": dict(stats['audio_properties']['sample_rates']),
                    "channels": dict(stats['audio_properties']['n_channels']),
                    "bit_depths": dict(stats['audio_properties']['bit_depths']),
                    "avg_file_size_kb": float(np.mean(stats['audio_properties']['file_sizes'])),
                    "audio_features": {
                        "zero_crossings": {
                            "mean": float(np.mean(stats['audio_properties']['zero_crossings'])),
                            "std": float(np.std(stats['audio_properties']['zero_crossings']))
                        },
                        "rms_energy": {
                            "mean": float(np.mean(stats['audio_properties']['rms_energy'])),
                            "std": float(np.std(stats['audio_properties']['rms_energy']))
                        },
                        "spectral_centroids": {
                            "mean": float(np.mean(stats['audio_properties']['spectral_centroids'])),
                            "std": float(np.std(stats['audio_properties']['spectral_centroids']))
                        },
                        "spectral_rolloffs": {
                            "mean": float(np.mean(stats['audio_properties']['spectral_rolloffs'])),
                            "std": float(np.std(stats['audio_properties']['spectral_rolloffs']))
                        }
                    }
                },
                "speaker_statistics": {
                    "total_speakers": len(stats['by_speaker']),
                    "gender_distribution": {
                        "male": sum(1 for s in stats['by_speaker'].values() if s['gender'] == 'Male'),
                        "female": sum(1 for s in stats['by_speaker'].values() if s['gender'] == 'Female')
                    },
                    "utterances_per_speaker": {
                        "mean": float(np.mean([data['count'] for data in stats['by_speaker'].values()])),
                        "median": float(np.median([data['count'] for data in stats['by_speaker'].values()])),
                        "min": float(min(data['count'] for data in stats['by_speaker'].values())),
                        "max": float(max(data['count'] for data in stats['by_speaker'].values()))
                    }
                }
            }
        }

        # Save to JSON file
        json_path = os.path.join(self.eda_dir, 'dataset_statistics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_stats, f, indent=4, ensure_ascii=False)
        
        # Print the statistics as before
        print("\n=== Dataset Statistics ===")
        print(f"\nTotal Files: {stats['total_files']}")
        
        print("\nEmotion Distribution:")
        for emotion, count in stats['by_emotion'].items():
            percentage = (count / stats['total_files']) * 100
            mean_duration = np.mean(stats['emotion_durations'][emotion])
            print(f"{emotion}: {count} files ({percentage:.1f}%) - Avg Duration: {mean_duration:.2f}s")
        
        print("\nGender Distribution:")
        for gender, count in stats['by_gender'].items():
            percentage = (count / stats['total_files']) * 100
            mean_duration = np.mean(stats['gender_durations'][gender])
            print(f"{gender}: {count} files ({percentage:.1f}%) - Avg Duration: {mean_duration:.2f}s")
        
        print("\nDuration Statistics:")
        print(f"Minimum duration: {stats['by_duration']['min']:.2f} seconds")
        print(f"Maximum duration: {stats['by_duration']['max']:.2f} seconds")
        print(f"Mean duration: {stats['by_duration']['mean']:.2f} seconds")
        print(f"Standard deviation: {stats['by_duration']['std']:.2f} seconds")
        print(f"Quartiles (25%, 50%, 75%): {stats['by_duration']['quartiles']}")
        
        print("\nAudio Properties:")
        print("Sample Rates:", dict(stats['audio_properties']['sample_rates']))
        print("Channels:", dict(stats['audio_properties']['n_channels']))
        print("Bit Depths:", dict(stats['audio_properties']['bit_depths']))
        print(f"Average File Size: {np.mean(stats['audio_properties']['file_sizes']):.2f} KB")
        
        print("\nSpeaker Statistics:")
        print(f"Total unique speakers: {len(stats['by_speaker'])}")
        male_speakers = sum(1 for s in stats['by_speaker'].values() if s['gender'] == 'Male')
        female_speakers = sum(1 for s in stats['by_speaker'].values() if s['gender'] == 'Female')
        print(f"Male speakers: {male_speakers}")
        print(f"Female speakers: {female_speakers}")
        
        # Calculate average utterances per speaker
        utterances_per_speaker = [data['count'] for data in stats['by_speaker'].values()]
        print(f"Average utterances per speaker: {np.mean(utterances_per_speaker):.2f}")
        print(f"Median utterances per speaker: {np.median(utterances_per_speaker):.2f}")
    
    def plot_statistics(self, stats: Dict[str, Any]) -> None:
        """Generate and save visualization plots"""
        # Set global font sizes
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 16,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 18
        })
        
        # Original plots
        self._plot_emotion_distribution(stats)
        self._plot_gender_distribution(stats)
        self._plot_duration_distribution(stats)
        self._plot_speaker_distribution(stats)
        self._plot_emotion_gender_distribution(stats)
        
        # New plots
        self._plot_duration_violin(stats)
        self._plot_audio_features_distribution(stats)
        self._plot_emotion_duration_box(stats)
        self._plot_speaker_emotion_heatmap(stats)
        self._plot_mfcc_distribution(stats)
    
    def _plot_emotion_distribution(self, stats):
        plt.figure(figsize=(10, 6))
        emotions = list(stats['by_emotion'].keys())
        counts = list(stats['by_emotion'].values())
        
        bars = plt.bar(emotions, counts)
        plt.title('Distribution of Emotions in Dataset')
        plt.xlabel('Emotion')
        plt.ylabel('Number of Files')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'emotion_distribution.png'))
        plt.close()
    
    def _plot_gender_distribution(self, stats):
        plt.figure(figsize=(8, 8))
        genders = list(stats['by_gender'].keys())
        counts = list(stats['by_gender'].values())
        
        plt.pie(counts, labels=genders, autopct='%1.1f%%',
                colors=['lightblue', 'lightpink'])
        plt.title('Gender Distribution in Dataset')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'gender_distribution.png'))
        plt.close()
    
    def _plot_duration_distribution(self, stats):
        plt.figure(figsize=(12, 8))
        plt.hist(stats['by_duration']['durations'], bins=30, edgecolor='black')
        plt.title('Distribution of Audio Durations', pad=20)
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Number of Files')
        
        plt.axvline(stats['by_duration']['mean'], color='r', linestyle='dashed',
                   linewidth=2, label=f"Mean ({stats['by_duration']['mean']:.2f}s)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'duration_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speaker_distribution(self, stats):
        plt.figure(figsize=(12, 6))
        speaker_counts = [data['count'] for data in stats['by_speaker'].values()]
        
        plt.hist(speaker_counts, bins=20, edgecolor='black')
        plt.title('Distribution of Utterances per Speaker')
        plt.xlabel('Number of Utterances')
        plt.ylabel('Number of Speakers')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'speaker_distribution.png'))
        plt.close()
    
    def _plot_emotion_gender_distribution(self, stats):
        # Prepare data
        emotions = list(stats['by_emotion'].keys())
        gender_emotion_counts = defaultdict(lambda: defaultdict(int))
        
        for speaker_id, speaker_data in stats['by_speaker'].items():
            gender = speaker_data['gender']
            for emotion, count in speaker_data['emotions'].items():
                gender_emotion_counts[gender][emotion] += count
        
        # Create grouped bar plot
        plt.figure(figsize=(14, 8))
        x = np.arange(len(emotions))
        width = 0.35
        
        male_counts = [gender_emotion_counts['Male'][emotion] for emotion in emotions]
        female_counts = [gender_emotion_counts['Female'][emotion] for emotion in emotions]
        
        plt.bar(x - width/2, male_counts, width, label='Male', color='lightblue')
        plt.bar(x + width/2, female_counts, width, label='Female', color='lightpink')
        
        plt.xlabel('Emotion')
        plt.ylabel('Number of Files')
        plt.title('Emotion Distribution by Gender', pad=20)
        plt.xticks(x, emotions)
        plt.legend(loc='upper right')
        
        # Add value labels on top of bars
        for i, count in enumerate(male_counts):
            plt.text(i - width/2, count, str(count), ha='center', va='bottom')
        for i, count in enumerate(female_counts):
            plt.text(i + width/2, count, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'emotion_gender_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_duration_violin(self, stats):
        plt.figure(figsize=(12, 6))
        
        # Prepare data for violin plot
        emotion_durations = {
            emotion: durations 
            for emotion, durations in stats['emotion_durations'].items()
        }
        
        # Create violin plot
        sns.violinplot(data=pd.DataFrame(emotion_durations))
        plt.title('Duration Distribution by Emotion')
        plt.xlabel('Emotion')
        plt.ylabel('Duration (seconds)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'duration_violin.png'))
        plt.close()
    
    def _plot_audio_features_distribution(self, stats):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Audio Features Distribution', y=1.02, fontsize=18)
        
        # Zero crossings
        sns.histplot(stats['audio_properties']['zero_crossings'], ax=axes[0,0])
        axes[0,0].set_title('Zero Crossings')
        axes[0,0].tick_params(labelsize=14)
        
        # RMS Energy
        sns.histplot(stats['audio_properties']['rms_energy'], ax=axes[0,1])
        axes[0,1].set_title('RMS Energy')
        axes[0,1].tick_params(labelsize=14)
        
        # Spectral Centroid
        sns.histplot(stats['audio_properties']['spectral_centroids'], ax=axes[1,0])
        axes[1,0].set_title('Spectral Centroid')
        axes[1,0].tick_params(labelsize=14)
        
        # Spectral Rolloff
        sns.histplot(stats['audio_properties']['spectral_rolloffs'], ax=axes[1,1])
        axes[1,1].set_title('Spectral Rolloff')
        axes[1,1].tick_params(labelsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'audio_features.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_emotion_duration_box(self, stats):
        plt.figure(figsize=(12, 6))
        
        # Prepare data for box plot
        emotion_durations = {
            emotion: durations 
            for emotion, durations in stats['emotion_durations'].items()
        }
        
        # Create box plot
        sns.boxplot(data=pd.DataFrame(emotion_durations))
        plt.title('Duration Distribution by Emotion (Box Plot)')
        plt.xlabel('Emotion')
        plt.ylabel('Duration (seconds)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'emotion_duration_box.png'))
        plt.close()
    
    def _plot_speaker_emotion_heatmap(self, stats):
        # Prepare data for heatmap
        speakers = list(stats['by_speaker'].keys())
        emotions = list(stats['by_emotion'].keys())
        
        data = np.zeros((len(speakers), len(emotions)))
        for i, speaker in enumerate(speakers):
            for j, emotion in enumerate(emotions):
                data[i,j] = stats['by_speaker'][speaker]['emotions'].get(emotion, 0)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(data, xticklabels=emotions, yticklabels=speakers, 
                    cmap='YlOrRd', annot=True, fmt='g', cbar_kws={'label': 'Count'})
        plt.title('Speaker-Emotion Distribution', pad=20)
        plt.xlabel('Emotion')
        plt.ylabel('Speaker ID')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'speaker_emotion_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_mfcc_distribution(self, stats):
        mfccs = np.array(stats['audio_properties']['mfccs'])
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=pd.DataFrame(mfccs))
        plt.title('MFCC Distribution')
        plt.xlabel('MFCC Coefficient')
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_dir, 'mfcc_distribution.png'))
        plt.close() 