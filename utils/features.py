import tensorflow as tf

def build_feature_columns():
    categorical_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list('feature_cat1', ['A', 'B', 'C']),
        tf.feature_column.categorical_column_with_vocabulary_list('feature_cat2', ['X', 'Y', 'Z'])
    ]
    
    # Wide部分：one-hot编码
    linear_feature_columns = [
        tf.feature_column.indicator_column(col) for col in categorical_columns
    ]

    # Deep部分：embedding + numeric
    dnn_feature_columns = [
        tf.feature_column.embedding_column(col, dimension=8) for col in categorical_columns
    ] + [
        tf.feature_column.numeric_column('feature_dense1'),
        tf.feature_column.numeric_column('feature_dense2')
    ]

    return linear_feature_columns, dnn_feature_columns
