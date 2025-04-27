import pandas as pd
import tensorflow as tf

def create_dataset(file_path, config, training=True):
    df = pd.read_csv(file_path)
    labels = df.pop(config['data']['label_column'])

    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if training:
        dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(config['train']['batch_size']).prefetch(tf.data.AUTOTUNE)
    return dataset
