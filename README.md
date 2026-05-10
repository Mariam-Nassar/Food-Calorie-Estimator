# Food-Calorie-Estimator

Deep Learning and Computer Vision project for estimating food calories from food images using the Nutrition5k dataset.

The project builds a complete end-to-end pipeline starting from raw image preprocessing to model training, evaluation, comparison, and deployment.

---

# Team Members

- Mariam Khalil
- Mahmoud Usama
- Omar Abdulaal

---

# Project Overview

Estimating food calories manually is difficult and time consuming.

This project uses Deep Learning techniques to automatically estimate the calorie value of food from a single image.

The system was built using:

- CNN architecture from scratch
- Transfer Learning using MobileNetV2
- Ensemble prediction approach
- Streamlit deployment

The project follows the full Computer Vision workflow required in the course guidelines.

---

# Dataset

Dataset Name:

Nutrition5k Dataset

Dataset Type:

Real-world food image dataset

Dataset Features:

- Food images
- Calorie values
- Nutrition information

Important Notes:

- The dataset is NOT a built-in dataset
- Images were processed manually
- Raw images were handled directly during preprocessing
- The dataset contains more than 1000 images

---

# Project Pipeline

## 1. Data Preparation

The preprocessing pipeline includes:

- Removing missing and corrupted images
- Downloading and organizing image files
- Image resizing
- Pixel normalization
- Train / Validation / Test split
- Data augmentation

### Applied Preprocessing

- Resize images to 224x224
- Normalize pixel values to [0,1]
- Random rotation
- Horizontal flip
- Zoom augmentation
- Brightness adjustment

Dataset Split:

- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

---

# Models

## 1. CNN From Scratch

A custom CNN architecture was built manually layer by layer using:

- Convolution Layers
- ReLU Activation
- MaxPooling
- Batch Normalization
- Dense Layers
- Dropout
- Regression Output Layer

This model was implemented completely from scratch without using pre-built classifiers.

---

## 2. Transfer Learning Model

Transfer Learning was implemented manually using MobileNetV2 as the backbone.

Implementation Steps:

- Loaded MobileNetV2 without the top classification head
- Added custom regression head manually
- Added Dense and Dropout layers explicitly
- Froze base layers during initial training
- Unfroze final layers for fine-tuning

The implementation follows the project requirement of building Transfer Learning manually instead of using drag-and-drop fine tuning.

---

# Model Training

Run the training notebook:

```bash
notebooks/02_model_training.ipynb
```

Or train directly from terminal:

```bash
python train_models.py
```

---

# Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# Quick Test

For quick testing with fewer epochs:

```bash
CNN_EPOCHS=1 TRANSFER_HEAD_EPOCHS=1 TRANSFER_FINE_TUNE_EPOCHS=1 python train_models.py
```

---

# Evaluation

This project predicts calorie values as continuous numerical outputs.

Therefore, the task is:

Regression Task

Because this is regression and NOT classification, evaluation uses regression metrics instead of classification accuracy metrics.

## Evaluation Metrics

- MAE
- RMSE
- R2 Score

The project also includes:

- Training Loss Curves
- Validation Loss Curves
- Model Comparison
- Prediction Visualization

---

# Current Results

| Model | MAE |
|---|---|
| CNN From Scratch | 90.96 |
| MobileNetV2 Transfer Learning | 87.27 |
| Ensemble Model | 77.26 |

Best Performing Model:

Ensemble Model

The latest comparison results are saved in:

```bash
results/model_comparison.csv
```

---

# Project Structure

```bash
Food-Calorie-Estimator/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_model_training.ipynb
│
├── models/
│
├── results/
│
├── app.py
├── train_models.py
├── download_images.py
├── requirements.txt
└── README.md
```

---

# Deployment

The project was deployed using Streamlit.

Live Demo:

https://food-calorie-sut.streamlit.app/

Features:

- Upload food image
- Predict calories
- Display prediction results
- Interactive interface

---

# How to Run the App

```bash
streamlit run app.py
```

---

# Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

---

# Conclusion

The project successfully demonstrates how Deep Learning and Computer Vision can estimate food calories automatically from images.

The Transfer Learning and Ensemble approaches achieved better performance than the CNN model built from scratch.

The deployed Streamlit application provides a complete real-world pipeline from image upload to calorie prediction.
