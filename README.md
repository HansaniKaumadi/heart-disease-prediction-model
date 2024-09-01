# Heart Disease Prediction Model

This project builds a heart disease prediction model using logistic regression and displays the output through a web application. The model predicts the likelihood of heart disease based on user-provided features, using a dataset from Kaggle.

## Project Overview

**Objective**: Develop a logistic regression model to predict heart disease and present the results via a user-friendly web interface.

**Dataset**: The dataset used is sourced from Kaggle. You can find it [here](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

## Features

- **Model Training**: Utilizes logistic regression for prediction based on heart disease indicators.
- **Web Interface**: Created with Streamlit to allow users to input feature values and receive predictions.
- **Visuals**: Includes an image to enhance the web interface.

## Technologies

- **Python**: Programming language used for developing the model and web application.
- **Libraries**:
  - `numpy` and `pandas` for data manipulation.
  - `scikit-learn` for model training and evaluation.
  - `streamlit` for building the web interface.
  - `PIL` (Pillow) for handling image display.
- **Dataset**: Kaggle heart disease dataset.

## Usage

- **Web App**: Once running, navigate to [http://localhost:8501](http://localhost:8501) in your web browser.
- **Input**: Enter comma-separated feature values (corresponding to the dataset's columns) in the provided text input box.
- **Prediction**: The web app will display whether the person has heart disease based on the input features.

## Model Performance

- **Training Data Accuracy**: Provides accuracy metrics and a classification report on the training dataset.
- **Test Data Accuracy**: Provides accuracy metrics and a classification report on the testing dataset.
