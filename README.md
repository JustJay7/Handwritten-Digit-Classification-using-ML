# Handwritten Digit Image Classification – ML-Powered Digit Recognition

This project is a machine learning-based system that classifies handwritten digits (0–9) using K-Nearest Neighbors (KNN), Artificial Neural Networks (ANN), and Random Forest (RF) models. Built on the MNIST dataset, it features an interactive image upload interface, visualized predictions, and performance metrics. It supports auto-saving models locally, making setup simple and self-contained.

---

## Features

- Supports Multiple Models – KNN, ANN, and Random Forest  
- Upload Interface – Predict digits from your own images  
- High Accuracy – 96–97%+ accuracy across models  
- Model Comparison – Evaluate metrics across KNN, ANN, and RF  
- Confusion Matrix Visualization  
- Auto-Saving Models – No manual download or upload needed  
- Interactive Predictions – Get visual confidence scores for digits

---

## Tech Stack

**Machine Learning:**  
Scikit-learn (KNN, Random Forest)  
TensorFlow/Keras (ANN)  
NumPy, Matplotlib  

**Dataset:**  
TensorFlow’s built-in MNIST dataset (28x28 grayscale images). There is no need to download the dataset, however, if you are curious what it looks like, you may download the dataset [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

---

## Setup Instructions

**Prerequisites**  
- Python 3.7 or higher  
- Git  
- pip (Python package manager)  

---

**Local Setup**

1. Clone the repository  
2. Install the required dependencies  
3. Run the main script

```bash
git clone https://github.com/JustJay7/Handwritten-Digit-Classification-using-ML.git
cd Handwritten-Digit-Classification-using-ML
pip install -r requirements.txt
python digit_classification_with_file_path.py
```

---

**Upon running the script:**

- The MNIST dataset is automatically downloaded using TensorFlow
- You’ll be prompted to select a model (KNN / ANN / RF)
- The selected model will be trained and saved locally as:
  - rf_model.pkl (Random Forest)
  - knn_model.pkl (KNN)
  - ann_model.h5 (ANN)

No manual dataset download or model upload is needed.

---

## How It Works

- The user selects one of the three models to train.  
- The MNIST dataset is loaded, preprocessed, and used to train the model.  
- The system calculates accuracy, precision, recall, F1 score, and confusion matrix.  
- The user can upload a grayscale digit image to see predictions with probability scores.  
- Visualizations are shown, and the model is saved for future use.

---

## Future Work

- Add CNN-based model for improved spatial accuracy  
- Support multi-digit recognition for handwritten numbers or zip codes  
- Deploy as a web application using Flask or Streamlit  

---

## Contributor

Jay Malhotra – BCA, Bennett University
