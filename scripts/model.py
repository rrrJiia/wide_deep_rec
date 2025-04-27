import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_wide_deep_model(feature_columns, config):
    # Build the inputs
    inputs = {col.name: keras.Input(name=col.name, shape=(), dtype=tf.float32) for col in feature_columns['dense_features']}
    inputs.update({col.name: keras.Input(name=col.name, shape=(), dtype=tf.string) for col in feature_columns['categorical_features']})

    # Wide part: linear model on categorical features
    wide_inputs = [feature_columns['categorical_feature_layer'](inputs)]
    wide_output = layers.Dense(1)(tf.concat(wide_inputs, axis=-1))

    # Deep part: DNN on embeddings + dense inputs
    deep_inputs = []
    for col in feature_columns['categorical_features']:
        embed = layers.Embedding(input_dim=col.vocabulary_size, output_dim=config['model']['embedding_dim'])(inputs[col.name])
        deep_inputs.append(layers.Flatten()(embed))

    for col in feature_columns['dense_features']:
        deep_inputs.append(tf.expand_dims(inputs[col.name], -1))

    deep_concat = layers.Concatenate()(deep_inputs)
    deep_output = layers.Dense(config['model']['deep_layer1'])(deep_concat)
    deep_output = layers.ReLU()(deep_output)
    deep_output = layers.Dense(config['model']['deep_layer2'])(deep_output)
    deep_output = layers.ReLU()(deep_output)

    # Final concat
    final_concat = layers.Concatenate()([wide_output, deep_output])
    output = layers.Dense(1, activation='sigmoid')(final_concat)

    model = keras.Model(inputs=inputs, outputs=output)
    return model
