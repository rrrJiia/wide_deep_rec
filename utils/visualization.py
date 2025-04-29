"""
Visualization utilities for Wide & Deep model.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_training_history(history, output_dir='logs'):
    """Plot training metrics over epochs.
    
    Args:
        history: History object returned by model.fit().
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot metrics
    metrics = ['loss', 'auc', 'precision', 'recall']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # Plot combined metrics in a single plot
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['auc', 'precision', 'recall']
    
    for metric in metrics_to_plot:
        plt.plot(history.history[metric], label=f'Train {metric}')
    
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
    plt.close()

def plot_feature_importance(model, feature_names, output_dir='logs'):
    """Plot feature importance based on model weights.
    
    Args:
        model: Trained TensorFlow model.
        feature_names: List of feature names.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the weights from the last layer
    weights = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.name == 'dense':
            weights = layer.get_weights()[0]
            break
    
    if len(weights) == 0:
        print("Could not extract feature importance from model weights.")
        return
    
    # Calculate feature importance (absolute values of weights)
    importance = np.mean(np.abs(weights), axis=1)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names[-len(importance):],
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    return importance_df

def plot_confusion_matrix(y_true, y_pred, output_dir='logs'):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted probabilities or labels.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert probabilities to binary predictions
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    elif y_pred.ndim > 1:
        y_pred = (y_pred > 0.5).astype(int).reshape(-1)
    else:
        y_pred = (y_pred > 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred, output_dir='logs'):
    """Plot ROC curve.
    
    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_prediction_distribution(y_pred, output_dir='logs'):
    """Plot distribution of predictions.
    
    Args:
        y_pred: Predicted probabilities.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten predictions if necessary
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=50, alpha=0.7, color='blue')
    plt.title('Distribution of Predictions')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    plt.close()

def visualize_model_performance(model, test_dataset, output_dir='logs'):
    """Comprehensive visualization of model performance.
    
    Args:
        model: Trained TensorFlow model.
        test_dataset: TensorFlow Dataset for testing.
        output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect predictions and true labels
    y_true = []
    y_pred = []
    
    for x, y in test_dataset:
        batch_pred = model.predict(x, verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend(batch_pred.flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Generate visualizations
    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_roc_curve(y_true, y_pred, output_dir)
    plot_prediction_distribution(y_pred, output_dir)
    
    # Create summary metrics
    binary_pred = (y_pred > 0.5).astype(int)
    accuracy = np.mean(binary_pred == y_true)
    
    # Save summary to text file
    with open(os.path.join(output_dir, 'performance_summary.txt'), 'w') as f:
        f.write("Model Performance Summary\n")
        f.write("========================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {auc(roc_curve(y_true, y_pred)[0], roc_curve(y_true, y_pred)[1]):.4f}\n")
        
        # Confusion matrix values
        cm = confusion_matrix(y_true, binary_pred)
        tn, fp, fn, tp = cm.ravel()
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
        
        # Additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f.write(f"\nPrecision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    print(f"Visualizations saved to {output_dir}")