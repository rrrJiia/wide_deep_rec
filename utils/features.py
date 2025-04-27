import tensorflow as tf

def build_feature_columns():
    # 假设有这两种特征
    categorical_features = [
        tf.feature_column.categorical_column_with_vocabulary_list('feature_cat1', ['A', 'B', 'C']),
        tf.feature_column.categorical_column_with_vocabulary_list('feature_cat2', ['X', 'Y', 'Z'])
    ]

    dense_features = [
        tf.feature_column.numeric_column('feature_dense1'),
        tf.feature_column.numeric_column('feature_dense2')
    ]

    # Wide部分直接用one-hot
    categorical_feature_layer = tf.keras.layers.DenseFeatures(
        [tf.feature_column.indicator_column(col) for col in categorical_features]
    )

    return {
        'categorical_features': categorical_features,
        'dense_features': dense_features,
        'categorical_feature_layer': categorical_feature_layer
    }
