import pandas as pd
import numpy as np
import os

def generate_data(num_samples, file_path):
    np.random.seed(42)

    data = {
        'feature_cat1': np.random.choice(['A', 'B', 'C'], size=num_samples),
        'feature_cat2': np.random.choice(['X', 'Y', 'Z'], size=num_samples),
        'feature_dense1': np.random.normal(loc=0.0, scale=1.0, size=num_samples),
        'feature_dense2': np.random.normal(loc=5.0, scale=2.0, size=num_samples),
    }

    # 假设label和特征简单线性相关
    logits = (
        (data['feature_cat1'] == 'A').astype(int) * 0.5 +
        (data['feature_cat2'] == 'X').astype(int) * 0.3 +
        data['feature_dense1'] * 0.2 +
        data['feature_dense2'] * 0.1
    )

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    labels = (probs > 0.5).astype(int)

    df = pd.DataFrame(data)
    df['label'] = labels

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved {num_samples} samples to {file_path}")

if __name__ == "__main__":
    generate_data(5000, 'data/train.csv')
    generate_data(1000, 'data/valid.csv')
    generate_data(1000, 'data/test.csv')
