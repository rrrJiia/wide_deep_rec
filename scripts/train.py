import os
import yaml
import pandas as pd
import tensorflow as tf
from utils.features import build_feature_columns
from utils.datasets import create_dataset
from scripts.model import build_wide_deep_model

# 1. 加载配置
if not os.path.exists('configs/config.yaml'):
    raise FileNotFoundError("配置文件 configs/config.yaml 未找到！")

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


train_data_path = config['data']['train_data_path']
valid_data_path = config['data']['valid_data_path']
test_data_path = config['data']['test_data_path']
batch_size = config['train']['batch_size']
epochs = config['train']['epochs']

hidden_units = config['model']['hidden_units']
dropout_rate = config['model']['dropout_rate']

# 2. 构建特征列
linear_feature_columns, dnn_feature_columns = build_feature_columns()

# 3. 创建数据集
# train_dataset = create_dataset(train_data_path, linear_feature_columns, dnn_feature_columns, batch_size, mode='train')
# valid_dataset = create_dataset(valid_data_path, linear_feature_columns, dnn_feature_columns, batch_size, mode='valid')
# test_dataset = create_dataset(test_data_path, linear_feature_columns, dnn_feature_columns, batch_size, mode='test')

train_dataset = create_dataset(train_data_path, config, training=True)
valid_dataset = create_dataset(valid_data_path, config, training=False)
test_dataset = create_dataset(test_data_path, config, training=False)

# 4. 创建模型
# model = build_wide_deep_model(linear_feature_columns, dnn_feature_columns, hidden_units=hidden_units, dropout_rate=dropout_rate)
model = build_wide_deep_model()
# 5. Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# 6. 训练模型
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    verbose=2
)

# 7. 测试集上评估
results = model.evaluate(test_dataset)
print(f"Test results - Loss: {results[0]}, AUC: {results[1]}, Precision: {results[2]}, Recall: {results[3]}")

# 8. 保存模型
save_dir = "saved_model"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, 'wide_deep_model.keras'))
print(f"Model saved to {save_dir}/wide_deep_model.keras")

