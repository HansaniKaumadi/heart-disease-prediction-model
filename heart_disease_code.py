#Date: 01/09/2024
#Coded by:Hansani Kaumadi
#A ML project to build a heart disease prediction model using logistic regression and display the output to the user through a website
#Dataset taken from kaggle URL: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset


#Importing the relevant libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import streamlit as st
from PIL import Image
import warnings
from sklearn.exceptions import ConvergenceWarning

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv(r'D:\ML projects\heart desease\heart.csv')

#understanding data 
print(heart_data.head()) # Display the first few rows of the dataset
print(heart_data.info()) # Display information about the dataset (e.g., column types, non-null counts)
print(heart_data.describe()) # Display basic statistics of the dataset
print(heart_data.shape) # Display the shape of the dataset (rows, columns)
print(heart_data.isna().sum()) # Display the count of missing values in each column

# Preparing the data for training the model
X = heart_data.drop(columns='target', axis=1)  # Features (excluding the target column)
Y = heart_data['target'] # Target variable (heart disease presence)
#print(X)
#print(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#Training the model using logistic regression 
model = LogisticRegression() # Initialize the logistic regression model
model.fit(X_train, Y_train) # Fit the model to the training data


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_data_accuracy)
print(classification_report(X_train_prediction, Y_train))


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(test_data_accuracy)
print(classification_report(X_test_prediction, Y_test))


#  Creating the web app
st.title('Heart Disease Prediction Model') # Set the title of the web app

# Input for prediction
input_text = st.text_input('Provide comma separated features to predict heart disease')
seperated_input = input_text.split(',') # Split the input text into a list of features
img = Image.open(r'D:\ML projects\heart desease\heart_img.png')
st.image(img,width=300) # Display the image on the web app

# Suppressing warnings related to convergence issues
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

# Making predictions based on user input
try:
    np_data = np.asarray(seperated_input,dtype=float)  # Convert the input list to a NumPy array
    reshaped_data = np_data.reshape(1,-1) # Reshape the array to match the model input
    prediction = model.predict(reshaped_data)
    if prediction[0] == 0:
        st.write("This person don't have a heart disease")
    else:
        st.write("this person have heart disease")

except ValueError:
    st.write('Invalid! Please provide comma seprated values correctly')

# Displaying additional information on the web app
st.subheader("About Data")
st.write(heart_data) # Display the dataset
st.subheader("Model Performance on Training Data")
st.write(training_data_accuracy) # Display the accuracy of the model on training data
st.subheader("Model Performance on Test Data")
st.write(test_data_accuracy)  # Display the accuracy of the model on test data
