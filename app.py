import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load the pre-trained soft voting ensemble model
model = joblib.load('soft_voting_model.pkl')
le = joblib.load('label_encoder.pkl')

# Helper function to preprocess the input image
def preprocess_image(image_bytes, size=(25, 25)):
    # Convert image to grayscale
    image = Image.open(image_bytes).convert("L")
    img = np.array(image)
    
    # Resize the image
    img_resized = cv2.resize(img, size)

    # Histogram Equalization
    hist_eq = cv2.equalizeHist(img_resized)

    # Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adapt_eq = clahe.apply(img_resized)

    # Flatten the images to pass into the model
    hist_eq_flat = hist_eq.flatten().reshape(1, -1)
    adapt_eq_flat = adapt_eq.flatten().reshape(1, -1)

    # Concatenate both feature sets (HE + AHE)
    combined_features = np.concatenate((hist_eq_flat, adapt_eq_flat), axis=1)
    
    return combined_features

# Streamlit UI
st.title('COVID-19 Prediction')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Debugging check for file uploader
if uploaded_file is not None:
    st.write("File uploaded successfully!")
    
    # Preprocess image
    features = preprocess_image(uploaded_file)

    # Make prediction using the pre-trained soft voting model
    prediction = model.predict(features)
    predicted_label = le.inverse_transform(prediction)[0]

    # Display prediction result
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    st.write(f"Prediction: {predicted_label}")
    
    # --- Ground Truth from filename ---
    if uploaded_file.name[0].lower() == 'c':
        st.write("Truth: Covid")
    else:
        st.write("Truth: Non-Covid")

else:
    st.write("Please upload an image to make a prediction.")
