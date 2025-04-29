import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load MovieLens CSVs
ratings = pd.read_csv('movie_raw_data/ratings.csv')
movies = pd.read_csv('movie_raw_data/movies.csv')
# Merge on movieId to attach genres to each rating
data = pd.merge(ratings, movies, on='movieId')
data['label'] = (data['rating'] >= 3.0).astype(int)
# Get all unique genres
all_genres = set()
for gen_list in data['genres']:
    for g in gen_list.split('|'):
        if g != "(no genres listed)":
            all_genres.add(g)
print(all_genres)
# Example output: {'Action', 'Adventure', 'Comedy', 'Drama', ...}
for genre in all_genres:
    data[f'genre_{genre}'] = data['genres'].str.contains(genre).astype(int)
data = data.drop(columns=['rating', 'timestamp', 'title', 'genres'])


train, temp = train_test_split(data, test_size=0.2, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)
print(len(train), len(valid), len(test))
file_path = "data/"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
# print(f"Saved {len(train)} samples to {file_path}")
# Save to CSV files that the code will read
train.to_csv('data/train.csv', index=False)
print(f"Saved {len(train)} samples to {file_path}")
valid.to_csv('data/valid.csv', index=False)
print(f"Saved {len(valid)} samples to {file_path}")
test.to_csv('data/test.csv', index=False)
print(f"Saved {len(test)} samples to {file_path}")
