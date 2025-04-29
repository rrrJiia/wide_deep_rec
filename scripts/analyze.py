"""
Analysis script for the Wide & Deep recommendation model.
This script analyzes model performance and generates detailed visualizations.
"""
import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, 
    auc, average_precision_score, confusion_matrix,
    classification_report
)

def load_saved_model(model_path):
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def load_test_data(test_data_path):
    """Load test data."""
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    
    test_df = pd.read_csv(test_data_path)
    print(f"Test data loaded from {test_data_path} ({len(test_df)} rows)")
    return test_df

def prepare_model_inputs(test_df, label_column):
    """Prepare inputs for model prediction."""
    # Separate features and labels
    test_labels = test_df[label_column].values
    
    # Prepare inputs
    test_inputs = {
        'userId': test_df['userId'].values.reshape(-1, 1),
        'movieId': test_df['movieId'].values.reshape(-1, 1)
    }
    
    # Add genre features
    for col in test_df.columns:
        if col.startswith('genre_'):
            test_inputs[col] = test_df[col].values.reshape(-1, 1)
    
    return test_inputs, test_labels

def get_predictions(model, test_inputs):
    """Get predictions from model."""
    predictions = model.predict(test_inputs, verbose=1)
    return predictions.flatten()

def compute_metrics(true_labels, predictions):
    """Compute comprehensive evaluation metrics."""
    # Calculate binary predictions
    binary_preds = (predictions > 0.5).astype(int)
    
    # Basic metrics
    metrics = {}
    
    # Classification report
    report = classification_report(true_labels, binary_preds, output_dict=True)
    metrics.update({
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score']
    })
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    metrics['roc_auc'] = auc(fpr, tpr)
    
    # PR AUC
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    metrics['pr_auc'] = auc(recall, precision)
    
    # Average precision
    metrics['avg_precision'] = average_precision_score(true_labels, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)
    
    return metrics

def plot_roc_curve(true_labels, predictions, output_dir):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

def plot_precision_recall_curve(true_labels, predictions, output_dir):
    """Plot precision-recall curve."""
    plt.figure(figsize=(8, 6))
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(true_labels, predictions)
    
    # Plot precision-recall curve
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (area = {pr_auc:.3f}, AP = {avg_precision:.3f})')
    
    # Plot baseline
    baseline = np.sum(true_labels) / len(true_labels)
    plt.plot([0, 1], [baseline, baseline], color='red', lw=2, linestyle='--', 
             label=f'Baseline (positive rate = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(true_labels, predictions, output_dir):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Calculate binary predictions
    binary_preds = (predictions > 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, binary_preds)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def plot_prediction_distribution(predictions, true_labels, output_dir):
    """Plot distribution of predictions by true label."""
    plt.figure(figsize=(10, 6))
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'prediction': predictions,
        'true_label': true_labels
    })
    
    # Plot histograms
    sns.histplot(data=df, x='prediction', hue='true_label', bins=50, 
                 palette={0: 'skyblue', 1: 'salmon'}, alpha=0.6)
    
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions by True Label')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300)
    plt.close()

def analyze_feature_importance_by_genre(test_df, predictions, label_column, output_dir):
    """Analyze feature importance by genre."""
    # Find genre columns
    genre_cols = [col for col in test_df.columns if col.startswith('genre_')]
    
    # Calculate correlation between genre and prediction
    importance = {}
    for col in genre_cols:
        # Correlation with predictions
        importance[col] = np.corrcoef(test_df[col], predictions)[0, 1]
    
    # Sort by absolute importance
    importance = {k: v for k, v in sorted(
        importance.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    
    # Prepare data for plotting
    genres = [g.replace('genre_', '') for g in importance.keys()]
    values = list(importance.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(genres))
    plt.barh(y_pos, values, color=colors)
    plt.yticks(y_pos, genres)
    plt.xlabel('Correlation with prediction')
    plt.title('Genre Correlation with Predictions')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'genre_importance.png'), dpi=300)
    plt.close()
    
    return importance

def generate_metrics_report(metrics, output_dir):
    """Generate a text report with metrics."""
    report_path = os.path.join(output_dir, 'model_performance.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("WIDE & DEEP RECOMMENDATION MODEL - PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CLASSIFICATION METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy:       {metrics['accuracy']:.4f}\n")
        f.write(f"Precision:      {metrics['precision']:.4f}\n")
        f.write(f"Recall:         {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:       {metrics['f1']:.4f}\n\n")
        
        f.write("RANKING METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"ROC AUC:        {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC:         {metrics['pr_auc']:.4f}\n")
        f.write(f"Avg Precision:  {metrics['avg_precision']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 30 + "\n")
        f.write(f"True Negatives:  {metrics['tn']}\n")
        f.write(f"False Positives: {metrics['fp']}\n")
        f.write(f"False Negatives: {metrics['fn']}\n")
        f.write(f"True Positives:  {metrics['tp']}\n\n")
        
        positive_rate = (metrics['tp'] + metrics['fn']) / sum([metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']])
        f.write(f"Positive Rate:   {positive_rate:.4f}\n\n")
        
        f.write("ADDITIONAL METRICS\n")
        f.write("-" * 30 + "\n")
        specificity = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
        f.write(f"Specificity:     {specificity:.4f}\n")
        
        # Model's decision threshold is 0.5
        f.write(f"Threshold:       0.5 (default)\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("End of Report\n")
    
    print(f"Metrics report saved to {report_path}")

def main():
    """Main analysis function."""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_path = 'saved_model/wide_deep_model.keras'
    model = load_saved_model(model_path)
    
    # Load test data
    test_data_path = config['data']['test_data_path']
    test_df = load_test_data(test_data_path)
    
    # Prepare inputs
    label_column = config['data']['label_column']
    test_inputs, test_labels = prepare_model_inputs(test_df, label_column)
    
    # Get predictions
    print("Getting predictions...")
    predictions = get_predictions(model, test_inputs)