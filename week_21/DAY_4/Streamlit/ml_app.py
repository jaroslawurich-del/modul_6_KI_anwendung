# -*- coding: utf-8 -*-
# app.py
#import streamlit as st

#st.title("Hello, Streamlit!")
#st.write("This is your first Streamlit app.")

#%%
# ml_app.py

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict the test set results
y_pred = clf.predict(X_test)

# Streamlit app
st.title("Iris Flower Prediction App")

# Create input fields for user to enter feature values
sepal_length = st.slider("Sepal Length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), float(X["sepal length (cm)"].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()), float(X["sepal width (cm)"].mean()))
petal_length = st.slider("Petal Length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), float(X["petal length (cm)"].mean()))
petal_width = st.slider("Petal Width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()), float(X["petal width (cm)"].mean()))

# Create a DataFrame for the input values
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)

# Make predictions
prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data)

# Display the prediction results
st.subheader("Prediction")
st.write(iris.target_names[prediction][0])
st.subheader("Prediction Probability")
st.write(prediction_proba)

