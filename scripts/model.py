# scripts/model.py

import tensorflow as tf
from utils.features import build_feature_columns

def build_wide_deep_model():
    wide_cols, deep_cols = build_feature_columns()
    
    feature_inputs = {}
    feature_inputs['userId'] = tf.keras.Input(shape=(), name='userId', dtype=tf.int32)
    feature_inputs['movieId'] = tf.keras.Input(shape=(), name='movieId', dtype=tf.int32)
    for genre_col in [col for col in deep_cols if col.__class__.__name__ == 'NumericColumn']:
        feature_inputs[genre_col.key] = tf.keras.Input(shape=(), name=genre_col.key, dtype=tf.float32)

    # Process Wide Inputs
    # Process Wide Inputs
    wide_inputs = []
    for col in wide_cols:
        if hasattr(col, 'categorical_column'):
            input_tensor = tf.keras.layers.Lambda(
                lambda x: tf.one_hot(x, depth=col.categorical_column.num_buckets)
            )(feature_inputs[col.categorical_column.key])
            wide_inputs.append(input_tensor)
        elif hasattr(col, 'key'):
            wide_inputs.append(tf.expand_dims(feature_inputs[col.key], -1))
    wide_feat_layer = tf.keras.layers.Concatenate()(wide_inputs)


    # Process Deep Inputs
    deep_inputs = []
    for col in deep_cols:
        if hasattr(col, 'categorical_column'):
            input_tensor = tf.keras.layers.Embedding(
                input_dim=col.categorical_column.num_buckets,
                output_dim=8)(feature_inputs[col.categorical_column.key])
            deep_inputs.append(input_tensor)
        elif hasattr(col, 'key'):
            deep_inputs.append(tf.expand_dims(feature_inputs[col.key], -1))
    deep_feat_layer = tf.concat(deep_inputs, axis=-1)

    # Deep MLP
    x = deep_feat_layer
    for units in [128, 64, 32]:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

    # Combine Wide and Deep
    combined_input = tf.keras.layers.concatenate([wide_feat_layer, x])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined_input)

    model = tf.keras.Model(inputs=feature_inputs, outputs=output)
    return model

