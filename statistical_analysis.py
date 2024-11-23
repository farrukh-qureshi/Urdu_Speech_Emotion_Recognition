import json
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class EmotionStatisticalAnalyzer:
    def __init__(self, json_path: str, output_dir: str = 'statistical_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Store emotions list
        self.emotions = list(self.data['dataset_summary']['emotion_distribution'].keys())
    
    def perform_anova_test(self, feature_data):
        """Perform one-way ANOVA test"""
        f_stat, p_value = stats.f_oneway(*feature_data)
        return f_stat, p_value
    
    def perform_tukey_test(self, data, emotion_labels):
        """Perform Tukey's HSD test"""
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        flat_data = np.concatenate(data)
        flat_labels = np.concatenate([[label] * len(d) for label, d in zip(emotion_labels, data)])
        tukey_results = pairwise_tukeyhsd(flat_data, flat_labels)
        return tukey_results
    
    def analyze_features(self):
        features = self.data['dataset_summary']['audio_properties']['audio_features']
        
        analysis_results = {
            'anova_results': {},
            'tukey_results': {},
            'effect_sizes': {}
        }
        
        # Create effect size matrix for all emotion pairs
        for feature_name, feature_stats in features.items():
            feature_data = []
            
            # Generate simulated data for each emotion
            for emotion in self.emotions:
                mean = feature_stats['mean']
                std = feature_stats['std']
                n_samples = self.data['dataset_summary']['emotion_distribution'][emotion]['count']
                emotion_data = np.random.normal(mean, std, n_samples)
                feature_data.append(emotion_data)
            
            # ANOVA test
            f_stat, p_value = self.perform_anova_test(feature_data)
            analysis_results['anova_results'][feature_name] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value)
            }
            
            # Tukey's test
            tukey_results = self.perform_tukey_test(feature_data, self.emotions)
            analysis_results['tukey_results'][feature_name] = str(tukey_results)
            
            # Effect size matrix
            n_emotions = len(self.emotions)
            effect_size_matrix = np.zeros((n_emotions, n_emotions))
            
            for i in range(n_emotions):
                for j in range(i+1, n_emotions):
                    d = self.cohens_d(feature_data[i], feature_data[j])
                    effect_size_matrix[i, j] = d
                    effect_size_matrix[j, i] = -d
            
            analysis_results['effect_sizes'][feature_name] = {
                'matrix': effect_size_matrix.tolist(),
                'emotions': self.emotions
            }
        
        return analysis_results
    
    @staticmethod
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se
    
    def visualize_results(self, results):
        # Create heatmaps for effect sizes
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
        with open(self.output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(results, f, indent=4)

def main():
    analyzer = EmotionStatisticalAnalyzer('EDA/dataset_statistics.json')
    results = analyzer.analyze_features()
    analyzer.visualize_results(results)
    
    # Print summary
    print("\nStatistical Analysis Summary:")
    print("\nANOVA Results:")
    for feature, result in results['anova_results'].items():
        print(f"\n{feature}:")
        print(f"F-statistic: {result['f_statistic']:.4f}")
        print(f"p-value: {result['p_value']:.4f}")
    
    print("\nEffect Sizes Summary:")
    for feature, effect_size_data in results['effect_sizes'].items():
        print(f"\n{feature}:")
        matrix = np.array(effect_size_data['matrix'])
        emotions = effect_size_data['emotions']
        for i in range(len(emotions)):
            for j in range(i+1, len(emotions)):
                print(f"{emotions[i]} vs {emotions[j]}: {matrix[i,j]:.4f}")

if __name__ == "__main__":
    main() 