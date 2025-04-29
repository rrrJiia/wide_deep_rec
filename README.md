# Wide & Deep Recommendation Model

This project implements a Wide & Deep model using TensorFlow/Keras, designed for basic structured data recommendation tasks.  
It includes data generation, feature engineering, model building, training, evaluation, and model saving.

---

## 📌 Project Structure

```
wide_deep_rec/
├── configs/                     # Base configuration files
│   └── config.yaml              # Default configuration
├── experiment_configs/          # Experiment-specific configurations
│   ├── baseline.yaml            # Default settings
│   ├── deep_network.yaml        # Deeper neural network
│   ├── high_dropout.yaml        # Higher regularization
│   └── ...                      # Other experiment configs
├── scripts/                     # Core implementation
│   ├── model.py                 # Wide & Deep model definition
│   ├── train.py                 # Training pipeline
│   ├── data_processor.py        # Data processing utilities
│   └── preprocess_original.py   # Original data preprocessing script
├── data/                        # Data directory
│   ├── train.csv                # Training data
│   ├── valid.csv                # Validation data 
│   └── test.csv                 # Test data
├── experiments/                 # Experiment outputs (auto-generated)
│   ├── baseline/                # Results for baseline experiment
│   ├── deep_network/            # Results for deep network experiment
│   └── ...                      # Other experiment results
├── run_experiments.py           # Main experiment runner
├── run.py                       # Single execution runner
```

---

## 🚀 Quick Start

### 1. Install Requirements

You should have Python 3.8+ installed.  
Recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

(*If `requirements.txt` not available, the key libraries are: `tensorflow`, `pandas`, `numpy`, `pyyaml`.*)

---

### 2. Generate Dataset

Synthetic datasets are automatically generated using:

```bash
python3 run.py --preprocess
```
(*By default, it preprocess the movie_raw_data and make them into the train/test/valid.csv files to be ready for train*)

---

### 3. Configure Training Parameters

Edit `configs/config.yaml` to adjust:

- Dataset paths
- Label column
- Batch size, number of epochs
- Model hidden layers and dropout rate

Example (`configs/config.yaml`):

```yaml
data:
  train_data_path: data/train.csv
  valid_data_path: data/valid.csv
  test_data_path: data/test.csv
  label_column: label

train:
  batch_size: 256
  epochs: 10

eval:
  batch_size: 512

model:
  hidden_units: [128, 64, 32]
  dropout_rate: 0.5
```
for experiment configs, please visit experiment_configs/
and change the files inside
---

### 4. Train and Evaluate

Simply run:

```bash
python3 -m scripts.train
```

- The model will train, validate, and evaluate on the test set.
- Metrics such as AUC, Precision, Recall, and Loss will be printed.
- The trained model will be saved in `saved_model/`.

---
Running Experiments:
Option 1: Run a single experiment
```bash
python run_experiments.py --experiment baseline
```
Available experiments include:
baseline
deep_network
high_dropout
small_batch
large_batch
no_dropout
sgd_optimizer
wide_only
deep_only
comedy_only
high_learning_rate
low_learning_rate
no_early_stopping
large_embeddings
minimal_epochs
---
Option 2: Run all experiments
```bash
python run_experiments.py --run-all
```
---
Option 3: Compare results
```bash
python run_experiments.py --compare
```

## 📈 Model Test Performance Example

| Metric | Value |
|:------|:------|
| Test Loss | ~0.079 |
| Test AUC | ~0.800 |
| Test Precision | ~0.983 |
| Test Recall | ~1.000 |

---

## Experiment Performance:

## Experiment Results

Our comprehensive experiments revealed significant performance variations across model configurations:

| Experiment          | AUC      | Precision | Recall   | Loss     |
|---------------------|:--------:|:---------:|:--------:|:--------:|
| small_batch         | **0.804**| 0.914     | 0.977    | **0.277**|
| high_learning_rate  | 0.801    | 0.901     | 0.995    | 0.278    |
| no_dropout          | 0.800    | 0.908     | 0.986    | 0.278    |
| deep_network        | 0.797    | 0.903     | 0.992    | 0.282    |
| baseline            | 0.797    | 0.897     | 0.997    | 0.283    |
| large_batch         | 0.790    | 0.905     | 0.989    | 0.285    |
| high_dropout        | 0.788    | 0.897     | 0.997    | 0.294    |
| low_learning_rate   | 0.752    | 0.893     | 1.000    | 0.302    |
| no_early_stopping   | 0.658    | 0.868     | 1.000    | 0.374    |
| large_embeddings    | 0.611    | 0.893     | 1.000    | 0.334    |
| sgd_optimizer       | 0.605    | 0.893     | 1.000    | 0.335    |
| deep_only           | 0.602    | 0.893     | 1.000    | 0.335    |
| minimal_epochs      | 0.602    | 0.893     | 1.000    | 0.335    |
| comedy_only         | 0.599    | 0.868     | 1.000    | 0.386    |
| wide_only           | 0.568    | 0.893     | 1.000    | 0.341    |

*Best values for each metric are in bold*

## 💬 Contact

If you have any questions or suggestions, feel free to reach out:

- **Author:** Ruoran Jia

---

## 📜 License

This project is licensed under the MIT License.  
Feel free to use, modify, and distribute with attribution.
