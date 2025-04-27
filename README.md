# Wide & Deep Recommendation Model

This project implements a Wide & Deep model using TensorFlow/Keras, designed for basic structured data recommendation tasks.  
It includes data generation, feature engineering, model building, training, evaluation, and model saving.

---

## 📌 Project Structure

```
wide_deep_rec/
├── configs/
│   └── config.yaml            # Training and evaluation configuration
├── data/
│   ├── train.csv               # Training data
│   ├── valid.csv               # Validation data
│   └── test.csv                # Testing data
├── scripts/
│   ├── model.py                # Wide & Deep model definition
│   └── train.py                # Training and evaluation pipeline
├── utils/
│   ├── datasets.py             # Dataset loading utilities
│   └── features.py             # Feature column definitions
├── saved_model/                # (Auto-generated) Exported models
├── checkpoints/                # (Optional) Model checkpoints during training
├── logs/                       # (Optional) TensorBoard logs
└── README.md                   # Project documentation
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
python3 scripts/generate_data.py
```
(*By default, it generates 5000 training samples, 1000 validation samples, and 1000 testing samples.*)

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

## 📈 Model Performance Example

| Metric | Value |
|:------|:------|
| Test Loss | ~0.079 |
| Test AUC | ~0.800 |
| Test Precision | ~0.983 |
| Test Recall | ~1.000 |

---

## 💬 Contact

If you have any questions or suggestions, feel free to reach out:

- **Author:** Carter He
- **Email:** [carterhes479@gmail.com]

---

## 📜 License

This project is licensed under the MIT License.  
Feel free to use, modify, and distribute with attribution.
