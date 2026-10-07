# 🌿 Plant Disease Detection Using CNN

A deep learning-based plant disease detection system that classifies plant leaf images into 38 different plant disease and healthy-leaf categories using a Convolutional Neural Network (CNN).

The trained model is deployed using Streamlit, allowing users to upload a plant leaf image and receive the top 3 predicted classes along with their confidence scores.

---

## 🚀 Features

- 🌱 Classifies plant leaf images into 38 classes
- 🧠 CNN model trained from scratch using TensorFlow/Keras
- 📊 Achieved 81.35% validation accuracy
- 🔍 Displays Top-3 predictions
- 📈 Shows confidence percentages
- 📷 Upload images directly through the web application
- 💻 Interactive Streamlit interface
- 🖼️ Supports JPG, JPEG and PNG images

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- Matplotlib
- Scikit-learn

---

## 📂 Dataset

This project uses the PlantVillage dataset.

The dataset contains images of healthy and diseased plant leaves covering 38 classes.

The dataset was divided into:

- 80% Training data
- 20% Validation data

The validation split was performed separately for each class to ensure that all 38 classes were represented in both training and validation sets.

---

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) trained from scratch.

### Model Pipeline

```text
Input Image (96 × 96 × 3)
        ↓
Data Augmentation
        ↓
Rescaling
        ↓
Convolution + Max Pooling
        ↓
Convolution + Max Pooling
        ↓
Convolution + Max Pooling
        ↓
Global Average Pooling
        ↓
Dense Layer
        ↓
Dropout
        ↓
38-Class Softmax Output