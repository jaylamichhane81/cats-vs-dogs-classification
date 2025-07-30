import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_vs_dogs_model.h5")

model = load_model()

# App title
st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload an image, and the model will predict whether it's a **Cat** or **Dog**.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((150, 150))  # Resize to match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    label = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")
