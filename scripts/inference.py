"""
Inference script for the Wide & Deep recommendation model.
"""
import os
import yaml
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import create_dataset

def load_model(model_path):
    """Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model.
        
    Returns:
        Loaded TensorFlow model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def predict_single_user_movie(model, user_id, movie_id, genre_features):
    """Make a prediction for a single user-movie pair.
    
    Args:
        model: Trained model.
        user_id: User ID.
        movie_id: Movie ID.
        genre_features: Dictionary of genre features {genre_name: 0/1}.
        
    Returns:
        Prediction probability.
    """
    # Create input dictionary
    input_dict = {
        'userId': np.array([user_id]),
        'movieId': np.array([movie_id])
    }
    
    # Add genre features
    for genre, value in genre_features.items():
        input_dict[genre] = np.array([value])
    
    # Make prediction
    prediction = model.predict(input_dict, verbose=0)
    return prediction[0][0]

def get_top_n_recommendations(model, user_id, movie_data, n=10):
    """Get top N movie recommendations for a user.
    
    Args:
        model: Trained model.
        user_id: User ID to get recommendations for.
        movie_data: DataFrame containing movie information.
        n: Number of recommendations to return.
        
    Returns:
        DataFrame with top N recommended movies.
    """
    # Make predictions for all movies for this user
    predictions = []
    
    for _, row in movie_data.iterrows():
        movie_id = row['movieId']
        
        # Extract genre features
        genre_features = {}
        for col in movie_data.columns:
            if col.startswith('genre_'):
                genre_features[col] = row[col]
        
        # Get prediction
        prediction = predict_single_user_movie(model, user_id, movie_id, genre_features)
        predictions.append((movie_id, prediction))
    
    # Sort by prediction score (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_movies = predictions[:n]
    
    # Create DataFrame
    result = pd.DataFrame({
        'movieId': [m[0] for m in top_movies],
        'score': [m[1] for m in top_movies]
    })
    
    # Merge with movie data to get additional info
    if 'title' in movie_data.columns:
        movie_info = movie_data[['movieId', 'title']]
        result = pd.merge(result, movie_info, on='movieId')
    
    return result

def predict_batch(model, test_data_path, config):
    """Make predictions on a batch of data.
    
    Args:
        model: Trained model.
        test_data_path: Path to test data.
        config: Configuration dictionary.
        
    Returns:
        DataFrame with predictions.
    """
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Create dataset
    test_dataset = create_dataset(test_data_path, config, training=False)
    
    # Make predictions
    predictions = model.predict(test_dataset)
    
    # Add predictions to dataframe
    test_df['prediction'] = predictions
    
    return test_df

def visualize_user_genre_preferences(recommendations, output_dir='logs'):
    """Visualize genre preferences for a user based on recommendations.
    
    Args:
        recommendations: DataFrame with recommended movies and genre features.
        output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract genre columns
    genre_cols = [col for col in recommendations.columns if col.startswith('genre_')]
    
    # Calculate average genre presence in recommendations
    genre_avg = {}
    for genre in genre_cols:
        # Weight by prediction score
        genre_avg[genre] = (recommendations[genre] * recommendations['score']).sum() / recommendations['score'].sum()
    
    # Sort genres by preference
    genre_avg = {k: v for k, v in sorted(genre_avg.items(), key=lambda item: item[1], reverse=True)}
    
    # Plot
    plt.figure(figsize=(12, 8))
    genres = list(genre_avg.keys())
    values = list(genre_avg.values())
    
    # Clean genre names for display
    clean_genres = [g.replace('genre_', '') for g in genres]
    
    sns.barplot(x=values, y=clean_genres)
    plt.title('User Genre Preferences')
    plt.xlabel('Preference Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'user_genre_preferences.png'))
    plt.close()

def main():
    """Main inference function."""
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = load_model('saved_model/wide_deep_model.keras')
    
    # Load a sample of test data
    test_df = pd.read_csv(config['data']['test_data_path'])
    
    # Example: Get predictions for the test set
    print("\nMaking batch predictions...")
    predictions_df = predict_batch(model, config['data']['test_data_path'], config)
    print(f"Predictions made for {len(predictions_df)} samples")
    
    # Example: Get recommendations for a specific user
    user_id = test_df['userId'].iloc[0]  # Use first user ID from test set
    print(f"\nGetting top 5 recommendations for user {user_id}...")
    recommendations = get_top_n_recommendations(model, user_id, test_df, n=5)
    print(recommendations)
    
    # Example: Make a single prediction
    movie_id = test_df['movieId'].iloc[0]  # Use first movie ID from test set
    
    # Extract genre features for this movie
    genre_features = {}
    for col in test_df.columns:
        if col.startswith('genre_'):
            genre_features[col] = test_df[col].iloc[0]
    
    print(f"\nPredicting for user {user_id} and movie {movie_id}...")
    prediction = predict_single_user_movie(model, user_id, movie_id, genre_features)
    print(f"Prediction: {prediction:.4f}")
    
    # Visualize user genre preferences
    print("\nVisualizing user genre preferences...")
    user_recommendations = get_top_n_recommendations(model, user_id, test_df, n=20)
    visualize_user_genre_preferences(user_recommendations)
    print("Visualization saved to logs/user_genre_preferences.png")

if __name__ == "__main__":
    main()