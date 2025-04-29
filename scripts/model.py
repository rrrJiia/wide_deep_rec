"""
Model definition for the Wide & Deep recommendation model.
Enhanced version with experimental options.
"""
import tensorflow as tf

def build_wide_deep_model(hidden_units=None, dropout_rate=0.5, embedding_dim=8, 
                         architecture='wide_deep', activation='relu'):
    """Build a Wide & Deep model for recommendation.
    
    Args:
        hidden_units: List of integers, the layer sizes of the DNN.
        dropout_rate: Float between 0 and 1, dropout rate for DNN layers.
        embedding_dim: Integer, dimension for embedding layers.
        architecture: String, model architecture type ('wide_deep', 'wide_only', 'deep_only').
        activation: String, activation function for hidden layers.
        
    Returns:
        A Keras Model instance.
    """
    if hidden_units is None:
        hidden_units = [128, 64, 32]
    
    # Define input layers
    userId_input = tf.keras.Input(shape=(1,), name='userId', dtype=tf.int32)
    movieId_input = tf.keras.Input(shape=(1,), name='movieId', dtype=tf.int32)
    
    # Define genre input layers
    genre_inputs = {}
    genre_input_list = []
    
    for genre in ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 
                 'Sci-Fi', 'Thriller', 'War', 'Western']:
        # Handle possible naming variations in the data
        genre_name = genre
        if genre == "Children":
            genre_name = "Children"  # Check actual name in your data
            
        input_name = f'genre_{genre_name}'
        genre_inputs[input_name] = tf.keras.Input(shape=(1,), name=input_name, dtype=tf.float32)
        genre_input_list.append(genre_inputs[input_name])
    
    # Concatenate all genre inputs for wide component
    genre_concat = tf.keras.layers.Concatenate()(genre_input_list)
    
    # Set user and movie ID ranges
    max_user_id = 200000  # Safely handle all user IDs
    max_movie_id = 200000  # Safely handle all movie IDs
    
    # Define activation function
    if activation == 'relu':
        activation_fn = 'relu'
    elif activation == 'leaky_relu':
        activation_fn = tf.keras.layers.LeakyReLU(alpha=0.2)
    elif activation == 'elu':
        activation_fn = 'elu'
    elif activation == 'tanh':
        activation_fn = 'tanh'
    elif activation == 'sigmoid':
        activation_fn = 'sigmoid'
    else:
        print(f"Warning: Unknown activation '{activation}', using ReLU instead.")
        activation_fn = 'relu'
    
    # Wide Component
    # User ID embedding for wide component (one-hot like)
    userId_wide = tf.keras.layers.Embedding(
        input_dim=max_user_id,
        output_dim=1,
        embeddings_initializer='zeros',
        trainable=True
    )(userId_input)
    userId_wide = tf.keras.layers.Flatten()(userId_wide)
    
    # Movie ID embedding for wide component (one-hot like)
    movieId_wide = tf.keras.layers.Embedding(
        input_dim=max_movie_id,
        output_dim=1,
        embeddings_initializer='zeros',
        trainable=True
    )(movieId_input)
    movieId_wide = tf.keras.layers.Flatten()(movieId_wide)
    
    # Combine wide inputs (linear model equivalent)
    wide = tf.keras.layers.Concatenate()([userId_wide, movieId_wide, genre_concat])
    wide = tf.keras.layers.Dense(1, activation=None, use_bias=True, name='wide_output')(wide)
    
    # Deep Component
    # User ID embedding for deep component
    userId_deep = tf.keras.layers.Embedding(
        input_dim=max_user_id,
        output_dim=embedding_dim,
        embeddings_initializer='uniform',
        trainable=True
    )(userId_input)
    userId_deep = tf.keras.layers.Flatten()(userId_deep)
    
    # Movie ID embedding for deep component
    movieId_deep = tf.keras.layers.Embedding(
        input_dim=max_movie_id,
        output_dim=embedding_dim,
        embeddings_initializer='uniform',
        trainable=True
    )(movieId_input)
    movieId_deep = tf.keras.layers.Flatten()(movieId_deep)
    
    # Combine embeddings and genre features
    deep = tf.keras.layers.Concatenate()([userId_deep, movieId_deep, genre_concat])
    
    # Add hidden layers for deep component
    for i, units in enumerate(hidden_units):
        # For string activations, we can pass directly to Dense
        if isinstance(activation_fn, str):
            deep = tf.keras.layers.Dense(units, activation=activation_fn, name=f'deep_dense_{i}')(deep)
        # For layer activations, we need to apply them after the Dense layer
        else:
            deep = tf.keras.layers.Dense(units, activation=None, name=f'deep_dense_{i}')(deep)
            deep = activation_fn(deep)
        
        deep = tf.keras.layers.Dropout(dropout_rate, name=f'deep_dropout_{i}')(deep)
    
    # Final dense layer for deep component
    deep = tf.keras.layers.Dense(1, activation=None, name='deep_output')(deep)
    
    # Build model based on architecture type
    if architecture == 'wide_only':
        output = tf.keras.layers.Activation('sigmoid')(wide)
        all_inputs = [userId_input, movieId_input] + list(genre_inputs.values())
        model = tf.keras.Model(inputs=all_inputs, outputs=output, name='wide_model')
    
    elif architecture == 'deep_only':
        output = tf.keras.layers.Activation('sigmoid')(deep)
        all_inputs = [userId_input, movieId_input] + list(genre_inputs.values())
        model = tf.keras.Model(inputs=all_inputs, outputs=output, name='deep_model')
    
    else:  # wide_deep (default)
        # Combine wide and deep components
        combined = tf.keras.layers.Add()([wide, deep])
        output = tf.keras.layers.Activation('sigmoid')(combined)
        
        # Create model with all inputs
        all_inputs = [userId_input, movieId_input] + list(genre_inputs.values())
        model = tf.keras.Model(inputs=all_inputs, outputs=output, name='wide_deep_model')
    
    return model