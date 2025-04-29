"""
Training script for the Wide & Deep recommendation model.
Enhanced version with support for experimental configurations.
"""
import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.model import build_wide_deep_model
from scripts.data_processor import load_data, preprocess_data, prepare_model_inputs

def train_and_visualize(config_path=None, output_dir=None):
    """Train the model and visualize results."""
    # 1. Load configuration
    if config_path is None:
        config_path = 'configs/config.yaml'
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found!")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Get configuration parameters
    train_data_path = config['data']['train_data_path']
    valid_data_path = config['data']['valid_data_path']
    test_data_path = config['data']['test_data_path']
    
    # Set up experiment parameters with defaults
    batch_size = config.get('train', {}).get('batch_size', 256)
    epochs = config.get('train', {}).get('epochs', 10)
    learning_rate = config.get('train', {}).get('learning_rate', 0.001)
    optimizer_type = config.get('train', {}).get('optimizer', 'adam')
    early_stopping = config.get('train', {}).get('early_stopping', True)
    patience = config.get('train', {}).get('patience', 3)
    momentum = config.get('train', {}).get('momentum', 0.9)  # For SGD
    
    hidden_units = config.get('model', {}).get('hidden_units', [128, 64, 32])
    dropout_rate = config.get('model', {}).get('dropout_rate', 0.5)
    embedding_dim = config.get('model', {}).get('embedding_dim', 8)
    architecture = config.get('model', {}).get('architecture', 'wide_deep')
    activation = config.get('model', {}).get('activation', 'relu')
    
    label_column = config.get('data', {}).get('label_column', 'label')
    genre_filter = config.get('data', {}).get('genre_filter', None)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.environ.get('EXPERIMENT_DIR', 'logs')
    
    # Create all subdirectories
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'saved_model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Print experiment configuration
    print(f"\nExperiment configuration:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Hidden units: {hidden_units}")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - Architecture: {architecture}")
    print(f"  - Optimizer: {optimizer_type}")
    print(f"  - Early stopping: {early_stopping}")
    
    # 3. Load and preprocess datasets
    print("\nLoading and preprocessing datasets...")
    
    # Load and preprocess train data
    train_df = load_data(train_data_path)
    train_df = preprocess_data(train_df, label_column)
    
    # Load and preprocess validation data
    valid_df = load_data(valid_data_path)
    valid_df = preprocess_data(valid_df, label_column)
    
    # Load and preprocess test data
    test_df = load_data(test_data_path)
    test_df = preprocess_data(test_df, label_column)
    
    # Apply genre filter if specified
    if genre_filter:
        print(f"Applying genre filter: {genre_filter}")
        train_df = train_df[train_df[genre_filter] == 1]
        valid_df = valid_df[valid_df[genre_filter] == 1]
        test_df = test_df[test_df[genre_filter] == 1]
    
    print(f"Train dataset: {train_df.shape}")
    print(f"Validation dataset: {valid_df.shape}")
    print(f"Test dataset: {test_df.shape}")
    
    # Prepare model inputs
    train_inputs, train_labels = prepare_model_inputs(train_df, label_column)
    valid_inputs, valid_labels = prepare_model_inputs(valid_df, label_column)
    test_inputs, test_labels = prepare_model_inputs(test_df, label_column)
    
    # 4. Create model
    print("Building Wide & Deep model...")
    model = build_wide_deep_model(
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        embedding_dim=embedding_dim,
        architecture=architecture,
        activation=activation
    )
    model.summary()
    
    # 5. Compile model
    print("Compiling model...")
    
    # Select optimizer based on configuration
    if optimizer_type.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_type.lower() == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    else:
        print(f"Warning: Unknown optimizer '{optimizer_type}', using Adam instead.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name='auc'), 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # 6. Create callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=output_dir,
            histogram_freq=1
        )
    ]
    
    # Add early stopping if enabled
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=patience,
                mode='max',
                restore_best_weights=True
            )
        )
    
    # Add model checkpoint
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model-{epoch:02d}-{val_auc:.4f}.keras'),
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
    )
    
    # 7. Train model
    print(f"Training model for {epochs} epochs...")
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
    model.save(os.path.join(model_dir, 'wide_deep_model.keras'))
    print(f"Model saved to {model_dir}/wide_deep_model.keras")
    
    # 10. Visualize results
    print("\nGenerating visualizations...")
    
    # Plot training history
    plot_training_history(history, output_dir=output_dir)
    
    # Visualize feature correlation importance
    visualize_feature_correlation_importance(train_df, label_column, output_dir=output_dir)
    
    # Generate predictions on test set for visualization
    test_pred = model.predict(test_inputs)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_pred, output_dir)
    
    # Plot ROC curve
    plot_roc_curve(test_labels, test_pred, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")
    
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
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300)
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
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
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Wide & Deep model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory for output files')
    
    args = parser.parse_args()
    
    train_and_visualize(config_path=args.config, output_dir=args.output_dir)