# Breast Cancer Detection using PyTorch

A simple Machine Learning / Deep Learning project built with PyTorch to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the Breast Cancer Wisconsin dataset.

This project demonstrates the complete ML workflow:
- Data preprocessing
- Feature scaling
- Label encoding
- Train-test split
- Converting NumPy arrays to PyTorch tensors
- Building a neural network
- Training with Gradient Descent
- Model evaluation

---

# Dataset

The dataset is loaded directly from GitHub using Pandas.

- Total Samples: **569**
- Features Used: **30**
- Target Variable:
  - `M` → Malignant
  - `B` → Benign

Dataset Source:
https://github.com/gscdit/Breast-Cancer-Detection

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch

---

# Project Workflow

## 1. Data Preprocessing
- Removed unnecessary columns:
  - `id`
  - `Unnamed: 32`
- Split features and labels
- Applied label encoding on diagnosis column

## 2. Feature Scaling
Used `StandardScaler()` to normalize the feature values.

## 3. Train-Test Split
Dataset split:
- 80% Training Data
- 20% Testing Data

## 4. Tensor Conversion
Converted NumPy arrays into PyTorch tensors for model training.

## 5. Neural Network Architecture

The model contains:
- One Linear Layer
- One Sigmoid Activation Function

Architecture:

```python
Linear(30 → 1)
Sigmoid()
```

---

# Training Details

| Parameter | Value |
|---|---|
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Loss Function | Binary Cross Entropy Loss |
| Epochs | 25 |

Loss gradually decreases during training, showing that the model learns effectively.

---

# Model Accuracy

Final Test Accuracy:

```python
Accuracy: 88.59%
```

---

# Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn
```

Run the notebook or Python script.

---

# Example Output

```python
Epoch: 1, Loss: 0.6661
Epoch: 10, Loss: 0.2442
Epoch: 20, Loss: 0.1789
Epoch: 25, Loss: 0.1624

Accuracy: 0.8859
```

---

# Future Improvements

- Add hidden layers for better performance
- Use Adam optimizer
- Add dropout regularization
- Implement confusion matrix visualization
- Deploy using Flask/FastAPI
- Create a frontend UI

---

# Learning Outcomes

This project helps understand:
- Fundamentals of PyTorch
- Binary Classification
- Neural Network Training Pipeline
- Tensor Operations
- Backpropagation
- Optimizers and Loss Functions

---

# Author

Made by Adarsh Anand
