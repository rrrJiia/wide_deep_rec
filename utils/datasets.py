import pandas as pd
import tensorflow as tf

def create_dataset(file_path, config, training=True):
    """Create a tf.data.Dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        config: Configuration dictionary.
        training: Whether this dataset is for training (to apply shuffling).
        
    Returns:
        A tf.data.Dataset object.
    """
    df = pd.read_csv(file_path)
    labels = df.pop(config['data']['label_column'])

    # Convert numerical columns to float32
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            # Keep userId and movieId as int32
            df[col] = df[col].astype('int32')

    # Convert label to int32
    labels = labels.astype('int32')

    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if training:
        dataset = dataset.shuffle(buffer_size=1024)

    # Set batch size based on mode
    batch_size = config['train']['batch_size']
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset