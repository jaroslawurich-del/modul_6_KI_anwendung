# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 13:48:28 2025

@author: milos
"""

import streamlit as st
from transformers import pipeline

# Cache the model loading to avoid reloading on each interaction
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

st.title("Sentence Classifier with Hugging Face")

# Input box for the user to enter a sentence
sentence = st.text_input("Enter a sentence for classification:")

# Button to trigger classification
if st.button("Classify"):
    if sentence.strip():
        result = model(sentence)
        label = result[0]['label']
        score = result[0]['score']
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {score:.2f}")
    else:
        st.warning("Please enter a sentence to classify.")
