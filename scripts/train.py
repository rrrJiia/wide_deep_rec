import tensorflow as tf
import yaml
from scripts.model import build_wide_deep_model
from utils.dataset import create_dataset
from utils.features import build_feature_columns

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_ds = create_dataset('data/processed/train.csv', config, training=True)
    test_ds = create_dataset('data/processed/test.csv', config, training=False)

    feature_columns = build_feature_columns()
    model = build_wide_deep_model(feature_columns, config)

    model.compile(optimizer=tf.keras.optimizers.Adam(config['train']['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])

    model.fit(train_ds, validation_data=test_ds, epochs=config['train']['epochs'])
    model.save('checkpoints/wide_deep_model')

if __name__ == '__main__':
    main()
