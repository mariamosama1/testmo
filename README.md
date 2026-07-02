# Brain Tumor MRI Classification

## Project Overview

This project classifies brain MRI images into one of four classes using Machine Learning and Deep Learning techniques. Three models were developed and compared: Random Forest, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP). The MLP achieved the best performance and was deployed in a Streamlit web application where users can upload an MRI image and receive a predicted tumor type.

---

## Dataset Description

**Dataset:** Brain Tumor Classification MRI Dataset

The dataset contains MRI brain images classified into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

### Preprocessing
- Images resized to 64 × 64 pixels
- Converted to grayscale
- Flattened into feature vectors (4096 features)
- Normalized and scaled before training

---

## Project Structure

```
Brain_Tumor_Project/
│
├── app.py                    # Streamlit application
├── BrainTumor.ipynb          # Kaggle/Jupyter notebook
├── brain_tumor_mlp.keras     # Trained MLP model
├── brain_tumor_rf.pkl        # Random Forest model
├── brain_tumor_svm.pkl       # SVM model
├── scaler.pkl                # StandardScaler
├── label_encoder.pkl         # Label Encoder
├── requirements.txt
└── README.md
```

---

## Installation Instructions

1. Clone the repository

```bash
git clone https://github.com/your_username/your_repository.git
```

2. Navigate to the project folder

```bash
cd your_repository
```

3. Install the required packages

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

Run the Streamlit application using:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

---

## Model Description

Three classification models were implemented and compared.

### Random Forest
- Ensemble machine learning algorithm
- Uses multiple decision trees
- Performance evaluated using Accuracy, Precision, Recall, and F1-Score

### Support Vector Machine (SVM)
- Uses the RBF kernel for classification
- Effective for high-dimensional image features
- Evaluated using Accuracy, Precision, Recall, and F1-Score

### Multi-Layer Perceptron (MLP)
- Deep learning neural network
- Input layer: 4096 features
- Hidden layer: 32 neurons (ReLU activation)
- Output layer: 4 neurons (Softmax activation)
- Trained using the Adam optimizer and Sparse Categorical Crossentropy loss

The MLP achieved the best overall performance and was selected for deployment in the Streamlit application.

---

## Streamlit Deployment

Live Demo:

https://your-streamlit-app-url.streamlit.app

---

## Author

Machine Learning Course Project