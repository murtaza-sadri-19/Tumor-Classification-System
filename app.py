import streamlit as st
import numpy as np
import joblib

# Load the ML model
model = joblib.load('Breast_cancer_model.pkl')

# Define the prediction function
def predict(features):
    features = np.array(features).reshape(1, -1)  # Reshape for a single prediction
    prediction = model.predict(features)
    return prediction[0]

# Required feature names
feature_names = [
    'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area',
    'Perimeter SE', 'Area SE', 'Worst Radius', 'Worst Texture',
    'Worst Perimeter', 'Worst Area', 'Worst Concavity'
]


# Streamlit app
st.title("Breast Tumour Classifier")

# Display the inputs in three columns
columns = st.columns(3)
features = []

for i, feature_name in enumerate(feature_names):
    with columns[i % 3]:  # This will alternate columns for each feature
        feature = st.number_input(feature_name, value=0.0, format="%.2f")
        features.append(feature)


# When the user clicks the button, make the prediction
if st.button("Predict"):
    # Predict using the input features
    prediction = predict(features)

    # Center align the prediction result
    st.markdown("<h3 style='text-align: center;'>Prediction Result:</h3>", unsafe_allow_html=True)
    
    # Display benign or malignant result
    if prediction == 0:
        st.markdown("<h2 style='text-align: center; color: red;'>Malignant Tumour</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: green;'>Benign Tumour</h2>", unsafe_allow_html=True)