# utils/features.py

import tensorflow as tf

def build_feature_columns():
    # 1. Categorical feature for userId
    user_col = tf.feature_column.categorical_column_with_identity(
        key='userId', num_buckets=611)  # assumes user IDs 1-610, bucket size = max_id+1
    # 2. Categorical feature for movieId
    movie_col = tf.feature_column.categorical_column_with_identity(
        key='movieId', num_buckets=10001)  # bucket slightly above max movieId (~9724)
    # 3. Indicator columns for wide part (one-hot representation)
    user_ind = tf.feature_column.indicator_column(user_col)
    movie_ind = tf.feature_column.indicator_column(movie_col)
    # 4. Numeric columns for each genre (already 0/1 in the data)
    genre_columns = []
    for genre in ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 
                  'Sci-Fi', 'Thriller', 'War', 'Western']:
        genre_columns.append(tf.feature_column.numeric_column(f'genre_{genre}', dtype=tf.float32))
    # 5. Wide feature set: user and movie indicators + genre flags
    wide_columns = [user_ind, movie_ind] + genre_columns
    # 6. Deep feature set: user & movie embeddings + genre flags
    user_emb = tf.feature_column.embedding_column(user_col, dimension=8)
    movie_emb = tf.feature_column.embedding_column(movie_col, dimension=8)
    deep_columns = [user_emb, movie_emb] + genre_columns
    return wide_columns, deep_columns
