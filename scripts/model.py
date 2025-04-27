# scripts/model.py

import tensorflow as tf

class WideDeepModel(tf.keras.Model):
    def __init__(self, linear_feature_columns, dnn_feature_columns, hidden_units=[128, 64, 32], dropout_rate=0.5):
        super(WideDeepModel, self).__init__()

        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        # --- Wide部分 ---
        # Dense特征直接连接线性层
        self.linear_dense = tf.keras.layers.Dense(1)

        # --- Deep部分 ---
        # 先加 StringLookup，把字符串类别 -> 整数 id
        self.lookup_layers = {}
        for feature in self.dnn_feature_columns:
            self.lookup_layers[feature['name']] = tf.keras.layers.StringLookup(
                vocabulary=feature['vocab_list'],
                output_mode='int',
                num_oov_indices=0  # 不允许OOV，避免漏掉类别导致崩溃
            )

        # 然后 Embedding：id -> embedding向量
        self.embedding_layers = {}
        for feature in self.dnn_feature_columns:
            self.embedding_layers[feature['name']] = tf.keras.layers.Embedding(
                input_dim=feature['vocab_size'],  # vocab_size必须对得上
                output_dim=feature['embedding_dim']
            )

        # 多层 DNN
        self.deep_dense_layers = []
        for units in self.hidden_units:
            self.deep_dense_layers.append(tf.keras.layers.Dense(units, activation='relu'))
            self.deep_dense_layers.append(tf.keras.layers.Dropout(self.dropout_rate))

        # 输出层
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # --- Wide部分 ---
        wide_inputs = tf.concat([
            tf.expand_dims(tf.cast(inputs[feature], tf.float32), axis=-1)
            for feature in self.linear_feature_columns
        ], axis=1)

        wide_out = self.linear_dense(wide_inputs)

        # --- Deep部分 ---
        embeddings = []
        for feature in self.dnn_feature_columns:
            raw_input = inputs[feature['name']]  # 原生string输入
            int_input = self.lookup_layers[feature['name']](raw_input)  # string -> int
            embed = self.embedding_layers[feature['name']](int_input)   # int -> embedding
            embeddings.append(embed)

        deep_inputs = tf.concat(embeddings, axis=-1)

        x = deep_inputs
        for layer in self.deep_dense_layers:
            x = layer(x, training=training)

        # --- 最后 Wide 和 Deep 合并 ---
        combined = tf.concat([wide_out, x], axis=-1)

        output = self.output_layer(combined)

        return output
