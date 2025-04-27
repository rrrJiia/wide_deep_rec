import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_wide_deep_model(feature_columns, config):
    # Build the inputs
    inputs = {col.name: keras.Input(name=col.name, shape=(), dtype=tf.float32) for col in feature_columns['dense_features']}
    inputs.update({col.name: keras.Input(name=col.name, shape=(), dtype=tf.string) for col in feature_columns['categorical_features']})

    # Wide part
    wide_output = layers.Dense(1)(feature_columns['categorical_feature_layer'](inputs))

    # Deep part
    deep_inputs = []
    for col in feature_columns['categorical_features']:
        input_dim = len(col.vocabulary_list) + 1
        embed = layers.Embedding(input_dim=input_dim, output_dim=config['model']['embedding_dim'])(inputs[col.name])
        deep_inputs.append(layers.Flatten()(embed))

    for col in feature_columns['dense_features']:
        deep_inputs.append(tf.expand_dims(inputs[col.name], -1))

    deep_concat = layers.Concatenate()(deep_inputs)

    x = deep_concat
    for units in config['hidden_units']:
        x = layers.Dense(units)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(config['dropout_rate'])(x)  # 可以加dropout防止过拟合
    deep_output = x

    # Final concat
    final_concat = layers.Concatenate()([wide_output, deep_output])
    output = layers.Dense(1, activation='sigmoid')(final_concat)

    model = keras.Model(inputs=inputs, outputs=output)
    return model
