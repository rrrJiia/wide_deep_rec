"""
Data processor for the Wide & Deep recommendation model.
This script handles data preprocessing and cleaning.
"""
import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

def process_movielens_data(ratings_path, movies_path, output_dir='data', random_state=42):
    """Process MovieLens data according to the original preprocessing logic.
    
    Args:
        ratings_path: Path to ratings.csv
        movies_path: Path to movies.csv
        output_dir: Directory to save processed data
        random_state: Random seed for train/test split
    
    Returns:
        Dictionary with paths to processed files
    """
    print(f"Loading MovieLens data from {ratings_path} and {movies_path}")
    
    # Load data
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Merge on movieId to attach genres to each rating
    data = pd.merge(ratings, movies, on='movieId')
    data['label'] = (data['rating'] >= 3.0).astype(int)

    positive_samples = data[data['label'] == 1]
    negative_samples = data[data['label'] == 0]
    
    # Option 1: Heavily imbalanced dataset (90% positive)
    pos_sample = positive_samples.sample(n=min(len(positive_samples), int(len(data)*0.9)), random_state=random_state)
    neg_sample = negative_samples.sample(n=min(len(negative_samples), int(len(data)*0.1)), random_state=random_state)
    data = pd.concat([pos_sample, neg_sample])
    
    # Get all unique genres
    all_genres = set()
    for gen_list in data['genres']:
        for g in gen_list.split('|'):
            if g != "(no genres listed)":
                all_genres.add(g)
    
    print(f"Found {len(all_genres)} unique genres: {sorted(all_genres)}")
    
    # Create binary indicators for each genre
    for genre in all_genres:
        data[f'genre_{genre}'] = data['genres'].str.contains(genre).astype(int)
    
    # Drop unnecessary columns
    data = data.drop(columns=['rating', 'timestamp', 'title', 'genres'])
    
    # Create train/validation/test split
    train, temp = train_test_split(data, test_size=0.2, random_state=random_state)
    valid, test = train_test_split(temp, test_size=0.5, random_state=random_state)
    
    print(f"Split data into {len(train)} train, {len(valid)} validation, and {len(test)} test samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV files
    train_path = os.path.join(output_dir, 'train.csv')
    valid_path = os.path.join(output_dir, 'valid.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"Saved processed data to {output_dir}/")
    
    return {
        'train': train_path,
        'valid': valid_path,
        'test': test_path
    }

def load_data(data_path, verbose=True):
    """Load data from CSV file and perform initial checks.
    
    Args:
        data_path: Path to CSV file
        verbose: Whether to print information
        
    Returns:
        DataFrame with loaded data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if verbose:
        print(f"Loaded {len(df)} rows from {data_path}")
        print(f"Columns: {df.columns.tolist()}")
    
    return df

def preprocess_data(df, label_column='label', verbose=True):
    """Preprocess the data for the Wide & Deep model.
    
    Args:
        df: DataFrame with raw data
        label_column: Name of the label column
        verbose: Whether to print information
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Check required columns
    required_columns = ['userId', 'movieId', label_column]
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get genre columns
    genre_columns = [col for col in processed_df.columns if col.startswith('genre_')]
    if not genre_columns:
        raise ValueError("No genre columns found (columns starting with 'genre_')")
    
    if verbose:
        print(f"Found {len(genre_columns)} genre columns")
    
    # Check for missing values
    missing_values = processed_df[required_columns + genre_columns].isna().sum()
    if missing_values.sum() > 0:
        print("Warning: Found missing values:")
        print(missing_values[missing_values > 0])
        
        # Fill missing values
        for col in required_columns:
            if col == 'userId' or col == 'movieId':
                # For IDs, we'll use a special 'unknown' ID (0)
                processed_df[col] = processed_df[col].fillna(0).astype(int)
            elif col == label_column:
                # For labels, we'll assume negative (0)
                processed_df[col] = processed_df[col].fillna(0).astype(int)
        
        # For genre columns, fill with 0 (not present)
        for col in genre_columns:
            processed_df[col] = processed_df[col].fillna(0).astype(float)
    
    # Convert userId and movieId to integers
    processed_df['userId'] = processed_df['userId'].astype(int)
    processed_df['movieId'] = processed_df['movieId'].astype(int)
    
    # Convert label to integer
    processed_df[label_column] = processed_df[label_column].astype(int)
    
    # Convert genre columns to float
    for col in genre_columns:
        processed_df[col] = processed_df[col].astype(float)
    
    # Check for out-of-range values
    if verbose:
        print(f"userId range: {processed_df['userId'].min()} to {processed_df['userId'].max()}")
        print(f"movieId range: {processed_df['movieId'].min()} to {processed_df['movieId'].max()}")
        
        # Check label distribution
        label_counts = processed_df[label_column].value_counts()
        print(f"Label distribution: {label_counts.to_dict()}")
    
    return processed_df

def prepare_model_inputs(df, label_column='label'):
    """Prepare the data for model training/evaluation.
    
    Args:
        df: Preprocessed DataFrame
        label_column: Name of the label column
        
    Returns:
        Tuple of (inputs_dict, labels)
    """
    # Extract labels
    labels = df[label_column].values
    
    # Prepare input dictionary
    inputs = {
        'userId': df['userId'].values.reshape(-1, 1),
        'movieId': df['movieId'].values.reshape(-1, 1)
    }
    
    # Add genre features
    for col in df.columns:
        if col.startswith('genre_'):
            inputs[col] = df[col].values.reshape(-1, 1)
    
    return inputs, labels

def split_train_test(df, test_size=0.2, random_state=42):
    """Split the data into training and test sets.
    
    Args:
        df: DataFrame to split
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def process_and_save_data(config_path='configs/config.yaml', overwrite=False):
    """Process all datasets and save the processed versions.
    
    Args:
        config_path: Path to configuration file
        overwrite: Whether to overwrite existing processed files
        
    Returns:
        Dictionary with paths to processed files
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if we should process original MovieLens data
    if 'raw_data' in config.get('data', {}):
        ratings_path = config['data']['raw_data'].get('ratings')
        movies_path = config['data']['raw_data'].get('movies')
        
        if ratings_path and movies_path and os.path.exists(ratings_path) and os.path.exists(movies_path):
            print(f"Found raw MovieLens data. Processing from original files...")
            processed_paths = process_movielens_data(ratings_path, movies_path)
            
            # Update config with processed paths
            config['data']['train_data_path'] = processed_paths['train']
            config['data']['valid_data_path'] = processed_paths['valid']
            config['data']['test_data_path'] = processed_paths['test']
            
            # Save updated config
            updated_config_path = 'configs/config_processed.yaml'
            with open(updated_config_path, 'w') as f:
                yaml.dump(config, f)
            
            return {
                'train': processed_paths['train'],
                'valid': processed_paths['valid'],
                'test': processed_paths['test'],
                'config': updated_config_path
            }
    
    # If we don't have raw data or should process existing CSVs
    # Get data paths from config
    train_path = config['data']['train_data_path']
    valid_path = config.get('data', {}).get('valid_data_path')
    test_path = config['data']['test_data_path']
    label_column = config['data']['label_column']
    
    # Create processed directory
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process train data
    train_df = load_data(train_path)
    processed_train_df = preprocess_data(train_df, label_column)
    processed_train_path = os.path.join(processed_dir, 'train_processed.csv')
    
    if overwrite or not os.path.exists(processed_train_path):
        processed_train_df.to_csv(processed_train_path, index=False)
        print(f"Saved processed training data to {processed_train_path}")
    
    # Process validation data if it exists
    processed_valid_path = None
    if valid_path:
        valid_df = load_data(valid_path)
        processed_valid_df = preprocess_data(valid_df, label_column)
        processed_valid_path = os.path.join(processed_dir, 'valid_processed.csv')
        
        if overwrite or not os.path.exists(processed_valid_path):
            processed_valid_df.to_csv(processed_valid_path, index=False)
            print(f"Saved processed validation data to {processed_valid_path}")
    
    # Process test data
    test_df = load_data(test_path)
    processed_test_df = preprocess_data(test_df, label_column)
    processed_test_path = os.path.join(processed_dir, 'test_processed.csv')
    
    if overwrite or not os.path.exists(processed_test_path):
        processed_test_df.to_csv(processed_test_path, index=False)
        print(f"Saved processed test data to {processed_test_path}")
    
    # Update config with processed paths
    config['data']['processed_train_data_path'] = processed_train_path
    if processed_valid_path:
        config['data']['processed_valid_data_path'] = processed_valid_path
    config['data']['processed_test_data_path'] = processed_test_path
    
    # Save updated config
    updated_config_path = 'configs/config_processed.yaml'
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Saved updated config to {updated_config_path}")
    
    return {
        'train': processed_train_path,
        'valid': processed_valid_path,
        'test': processed_test_path,
        'config': updated_config_path
    }

if __name__ == "__main__":
    process_and_save_data(overwrite=True)