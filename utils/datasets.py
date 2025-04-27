import pandas as pd
import tensorflow as tf

def create_dataset(file_path, config, training=True):
    df = pd.read_csv(file_path)
    labels = df.pop(config['data']['label_column'])

    # 强制数值列转float32
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].astype('float32')

    # label转成int32
    labels = labels.astype('int32')

    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if training:
        dataset = dataset.shuffle(buffer_size=1024)

    # 训练和验证/测试可以设置不同batch_size
    batch_size = config['train']['batch_size']
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
