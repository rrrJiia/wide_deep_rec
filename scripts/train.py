import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.model import build_wide_deep_model
from scripts.data_processor import load_data, preprocess_data, prepare_model_inputs

def train_and_visualize():
    """Train the model and visualize results."""
    # 1. Load configuration
    if not os.path.exists('configs/config.yaml'):
        raise FileNotFoundError("Configuration file configs/config.yaml not found!")

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Get configuration parameters
    train_data_path = config['data']['train_data_path']
    valid_data_path = config['data']['valid_data_path']
    test_data_path = config['data']['test_data_path']
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    hidden_units = config['model']['hidden_units']
    dropout_rate = config['model']['dropout_rate']
    label_column = config['data']['label_column']

    # 3. Load and preprocess datasets
    print("Loading and preprocessing datasets...")
    
    # Load and preprocess train data
    train_df = load_data(train_data_path)
    train_df = preprocess_data(train_df, label_column)
    
    # Load and preprocess validation data
    valid_df = load_data(valid_data_path)
    valid_df = preprocess_data(valid_df, label_column)
    
    # Load and preprocess test data
    test_df = load_data(test_data_path)
    test_df = preprocess_data(test_df, label_column)
    
    print(f"Train dataset: {train_df.shape}")
    print(f"Validation dataset: {valid_df.shape}")
    print(f"Test dataset: {test_df.shape}")
    
    # Prepare model inputs
    train_inputs, train_labels = prepare_model_inputs(train_df, label_column)
    valid_inputs, valid_labels = prepare_model_inputs(valid_df, label_column)
    test_inputs, test_labels = prepare_model_inputs(test_df, label_column)
    
    # 4. Create model
    print("Building Wide & Deep model...")
    model = build_wide_deep_model(hidden_units=hidden_units, dropout_rate=dropout_rate)
    model.summary()
    
    # 5. Compile model
    print("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name='auc'), 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # 6. Create callbacks for model checkpoints and early stopping
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model-{epoch:02d}-{val_auc:.4f}.keras'),
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    # 7. Train model
    print(f"Training model for {epochs} epochs (with early stopping)...")
    history = model.fit(
        train_inputs,
        train_labels,
        validation_data=(valid_inputs, valid_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Evaluate on test set
    print("\nEvaluating model on test set...")
    results = model.evaluate(test_inputs, test_labels, batch_size=batch_size)
    print(f"Test results - Loss: {results[0]:.4f}, AUC: {results[1]:.4f}, "
          f"Precision: {results[2]:.4f}, Recall: {results[3]:.4f}")

    # 9. Save the model
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'wide_deep_model.keras'))
    print(f"Model saved to {save_dir}/wide_deep_model.keras")
    
    # 10. Visualize results
    print("\nGenerating visualizations...")
    
    # Plot training history
    plot_training_history(history, output_dir=log_dir)
    
    # Visualize feature correlation importance
    visualize_feature_correlation_importance(train_df, label_column, output_dir=log_dir)
    
    # Generate predictions on test set for visualization
    test_pred = model.predict(test_inputs)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_pred, log_dir)
    
    # Plot ROC curve
    plot_roc_curve(test_labels, test_pred, log_dir)
    
    print(f"All visualizations saved to {log_dir}/")
    
    return history, results, model

def plot_training_history(history, output_dir='logs'):
    """Plot training metrics over epochs."""
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

def visualize_feature_correlation_importance(df, label_column, output_dir='logs'):
    """Visualize feature importance based on correlation with target."""
    # Find genre columns
    genre_cols = [col for col in df.columns if col.startswith('genre_')]
    
    # Calculate correlation-based importance
    importance = {}
    for col in genre_cols:
        importance[col] = abs(df[col].corr(df[label_column]))
    
    # Sort by importance
    importance = {k: v for k, v in sorted(
        importance.items(), key=lambda item: item[1], reverse=True)}
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    
    genres = list(importance.keys())
    values = list(importance.values())
    
    # Clean genre names for display (remove 'genre_' prefix)
    clean_genres = [g.replace('genre_', '') for g in genres]
    
    # Create a color palette based on importance
    colors = sns.color_palette('viridis', len(genres))
    
    plt.barh(clean_genres, values, color=colors)
    plt.xlabel('Correlation with Target Label', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.title('Genre Importance for Movie Recommendations', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'feature_correlation_importance.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir='logs'):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred, output_dir='logs'):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
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
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

if __name__ == "__main__":
    train_and_visualize()