# utils/features.py

def build_feature_columns():
    # Wide特征（数值特征）
    linear_feature_columns = [
        'feature_dense1',
        'feature_dense2'
    ]

    # Deep特征（类别特征，需要Embedding）
    dnn_feature_columns = [
        {
            'name': 'feature_cat1',
            'vocab_size': 3,            # A/B/C 三个类别
            'embedding_dim': 4,          # 自己随便选的，可以调整
            'vocab_list': ['A', 'B', 'C']
        },
        {
            'name': 'feature_cat2',
            'vocab_size': 3,            # X/Y/Z 三个类别
            'embedding_dim': 4,
            'vocab_list': ['X', 'Y', 'Z']
        }
    ]

    return linear_feature_columns, dnn_feature_columns
