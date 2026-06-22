"""
Module: calculate_metrics.py
Description: 
    Calculates academic metrics dynamically. Automatically detects 
    any generated threat category probabilities and aggregates them.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MetricsConfig:
    CSV_PATH = "results_images_2.csv"  # Path to the CSV containing predictions and ground truth
    EVAL_THRESHOLDS = [55.0, 65.0, 85.0]

def map_ground_truth(label):
    label = str(label)
    if 'Unsafe' in label: return 1
    if 'Safe' in label: return 0
    return None

def generate_metric_reports():
    if not os.path.exists(MetricsConfig.CSV_PATH):
        print(f"Error: {MetricsConfig.CSV_PATH} not found.")
        return

    df = pd.read_csv(MetricsConfig.CSV_PATH)
    df = df[df['ground_truth_label'] != 'Unknown'].copy()
    df['y_true'] = df['ground_truth_label'].apply(map_ground_truth)
    df = df.dropna(subset=['y_true'])

    prob_columns = [col for col in df.columns if col.endswith('_prob')]
    
    if not prob_columns:
        print("Error: No probability columns found in CSV.")
        return

    fig, axes = plt.subplots(1, len(MetricsConfig.EVAL_THRESHOLDS), figsize=(18, 5))
    print("="*50)
    print(f"Dynamic Policy Evaluation ({len(prob_columns)} Threats detected)")
    print("="*50)

    for i, thresh in enumerate(MetricsConfig.EVAL_THRESHOLDS):
        df[f'y_pred_{thresh}'] = (df[prob_columns] >= thresh).any(axis=1).astype(int)
        
        y_true = df['y_true'].tolist()
        y_pred = df[f'y_pred_{thresh}'].tolist()
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"Threshold: {thresh}%")
        print(f"  Accuracy  : {acc * 100:.2f}%")
        print(f"  Precision : {prec * 100:.2f}%")
        print(f"  Recall    : {rec * 100:.2f}%")
        print(f"  F1-Score  : {f1 * 100:.2f}%\n")
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred SAFE', 'Pred BLOCKED'],
                    yticklabels=['Actual SAFE', 'Act BLOCKED'],
                    annot_kws={"size": 14, "weight": "bold"})
        
        axes[i].set_title(f'Threshold {thresh}%\nAccuracy: {acc*100:.1f}%', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('dynamic_thresholds_evaluation.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to 'dynamic_thresholds_evaluation.png'.")

if __name__ == "__main__":
    generate_metric_reports()