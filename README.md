# 🌿 Plant Disease Detection Using CNN

An AI-powered web application that detects plant leaf diseases from images using a **Convolutional Neural Network (CNN)** trained from scratch.

The model is trained on the **PlantVillage dataset** and can classify plant leaves into **38 different disease/healthy classes**.

## 🚀 Live Demo

👉 [🌿 Plant Disease Detection – Live App](https://plant-disease-detection-dappua69fenfnuj4l97gn3m.streamlit.app/)

## 📌 Project Overview

Plant diseases can significantly affect crop production and quality. Early detection can help identify diseases before they spread and cause major damage.

This project uses **Deep Learning and Computer Vision** to automatically analyze a plant leaf image and predict its disease class.

The trained CNN model is integrated into a **Streamlit web application**, allowing users to upload a leaf image and receive a prediction with confidence scores.

## ✨ Features

- 🌿 Plant leaf disease classification
- 🧠 CNN model trained from scratch
- 📚 38 disease and healthy classes
- 📊 81.35% validation accuracy
- 🔍 Top-3 prediction results
- 📈 Confidence scores
- 🖼️ Image upload through web interface
- 🌐 Live Streamlit deployment
- 💻 Simple and user-friendly interface

## 🛠️ Technologies Used

### Programming
- Python

### Machine Learning / Deep Learning
- TensorFlow
- Keras
- Convolutional Neural Network (CNN)

### Data Processing
- NumPy
- Pillow

### Web Application
- Streamlit

### Development Tools
- VS Code
- Git
- GitHub

## 📂 Dataset

The project uses the **PlantVillage dataset** for training and validation.

The dataset contains images belonging to **38 plant disease and healthy classes**.

The dataset is not included in this repository because of its large size.

## 🧠 Model Architecture

The CNN model was trained from scratch using the following architecture:

```text
Input Image (96 × 96 × 3)
        ↓
Data Augmentation
        ↓
Rescaling
        ↓
Conv2D (32 filters)
        ↓
MaxPooling
        ↓
Conv2D (64 filters)
        ↓
MaxPooling
        ↓
Conv2D (128 filters)
        ↓
MaxPooling
        ↓
Global Average Pooling
        ↓
Dense (128)
        ↓
Dropout (0.4)
        ↓
Softmax Output
        ↓
38 Classes