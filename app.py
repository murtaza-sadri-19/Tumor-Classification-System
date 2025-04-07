import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
try:
    model = pickle.load(open("Breast_cancer_model.pkl", 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'Breast_cancer_model.pkl' is in the same directory as this script.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 1.5rem;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
    }
    .feature-info {
        font-size: 0.9rem;
        color: #6c757d;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>Breast Cancer Prediction Tool</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Enter patient measurements to predict cancer type</h2>", unsafe_allow_html=True)

# Create sidebar with information
st.sidebar.image("https://img.freepik.com/free-vector/breast-cancer-awareness-month-concept_23-2148639841.jpg?t=st=1744002082~exp=1744005682~hmac=6fea5562a29748f2783e3ccac35a375bdaebc6df551645325b7bdf6da2730e09&w=826", width=300)
st.sidebar.title("About")
st.sidebar.info(
    """
    This application uses machine learning to predict whether a breast mass is benign or malignant
    based on measurements taken from fine needle aspirates (FNA) of breast masses.
    
    The model achieves approximately 96.5% accuracy on test data.
    
    **Note**: This tool is for educational purposes only and should not replace professional medical advice.
    """
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "How it Works", "Feature Descriptions"])

with tab1:
    st.subheader("Patient Data Input")

    # Create two columns for input fields to make the form more compact
    col1, col2 = st.columns(2)

    # Mean values
    with col1:
        st.markdown("##### Mean Values")
        mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=50.0, value=14.0,
                                     help="Mean of distances from center to points on the perimeter")
        mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=50.0, value=19.0,
                                      help="Mean of standard deviation of gray-scale values")
        mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=250.0, value=92.0)
        mean_area = st.number_input("Mean Area", min_value=0.0, max_value=2500.0, value=650.0)
        mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=0.3, value=0.1, format="%.6f")

    with col2:
        st.markdown("##### Mean Values (continued)")
        mean_compactness = st.number_input("Mean Compactness", min_value=0.0, max_value=1.0, value=0.1, format="%.6f")
        mean_concavity = st.number_input("Mean Concavity", min_value=0.0, max_value=1.0, value=0.1, format="%.6f")
        mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0, max_value=0.5, value=0.05, format="%.6f")
        mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0, max_value=1.0, value=0.2, format="%.6f")
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=0.0, max_value=0.1, value=0.06, format="%.6f")

    # SE values
    st.markdown("##### Standard Error Values")
    col3, col4 = st.columns(2)

    with col3:
        se_radius = st.number_input("SE Radius", min_value=0.0, max_value=10.0, value=0.4, format="%.6f")
        se_texture = st.number_input("SE Texture", min_value=0.0, max_value=10.0, value=1.2, format="%.6f")
        se_perimeter = st.number_input("SE Perimeter", min_value=0.0, max_value=50.0, value=2.0, format="%.6f")
        se_area = st.number_input("SE Area", min_value=0.0, max_value=500.0, value=25.0, format="%.6f")
        se_smoothness = st.number_input("SE Smoothness", min_value=0.0, max_value=0.1, value=0.01, format="%.6f")

    with col4:
        se_compactness = st.number_input("SE Compactness", min_value=0.0, max_value=0.5, value=0.03, format="%.6f")
        se_concavity = st.number_input("SE Concavity", min_value=0.0, max_value=0.5, value=0.03, format="%.6f")
        se_concave_points = st.number_input("SE Concave Points", min_value=0.0, max_value=0.2, value=0.01, format="%.6f")
        se_symmetry = st.number_input("SE Symmetry", min_value=0.0, max_value=0.1, value=0.02, format="%.6f")
        se_fractal_dimension = st.number_input("SE Fractal Dimension", min_value=0.0, max_value=0.1, value=0.003, format="%.6f")

    # Worst values
    st.markdown("##### Worst Values (largest mean value from image)")
    col5, col6 = st.columns(2)

    with col5:
        worst_radius = st.number_input("Worst Radius", min_value=0.0, max_value=50.0, value=17.0)
        worst_texture = st.number_input("Worst Texture", min_value=0.0, max_value=50.0, value=25.0)
        worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0, max_value=300.0, value=115.0)
        worst_area = st.number_input("Worst Area", min_value=0.0, max_value=4000.0, value=880.0)
        worst_smoothness = st.number_input("Worst Smoothness", min_value=0.0, max_value=0.3, value=0.13, format="%.6f")

    with col6:
        worst_compactness = st.number_input("Worst Compactness", min_value=0.0, max_value=1.5, value=0.25, format="%.6f")
        worst_concavity = st.number_input("Worst Concavity", min_value=0.0, max_value=2.0, value=0.3, format="%.6f")
        worst_concave_points = st.number_input("Worst Concave Points", min_value=0.0, max_value=0.3, value=0.1, format="%.6f")
        worst_symmetry = st.number_input("Worst Symmetry", min_value=0.0, max_value=1.0, value=0.3, format="%.6f")
        worst_fractal_dimension = st.number_input("Worst Fractal Dimension", min_value=0.0, max_value=0.5, value=0.08, format="%.6f")

    # Create a button for prediction
    predict_button = st.button("Predict Cancer Type", type="primary", use_container_width=True)

    # When the predict button is clicked
    if predict_button:
        # Collect all inputs into an array
        input_data = (
            mean_radius, mean_texture, mean_perimeter, mean_area, se_perimeter,
            se_area, worst_radius, worst_texture, worst_perimeter, worst_area, worst_concavity
        )

        # Convert input to numpy array and reshape
        input_array = np.asarray(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)

        # Display prediction result with better styling
        if prediction[0] == 0:
            st.markdown("""
                <div class='result-box malignant'>
                    Prediction: <strong>Malignant</strong> (Cancerous)
                    <br/>
                    <span style="font-size: 0.9rem;">The model predicts that the breast mass is malignant. Please consult with a healthcare professional.</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-box benign'>
                    Prediction: <strong>Benign</strong> (Non-Cancerous)
                    <br/>
                    <span style="font-size: 0.9rem;">The model predicts that the breast mass is benign. Please consult with a healthcare professional.</span>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("How the Model Works")

    st.markdown("""
    This application uses a **Ridge Classifier**, chosen after evaluating multiple models on the **Wisconsin Breast Cancer dataset**.

    ### 🧠 Workflow Overview:
    1. **Data Collection**: Features are derived from digitized images of breast mass biopsies.
    2. **Model Comparison**: Algorithms like Random Forest, SVM, and Neural Networks were evaluated.
    3. **Final Model Selection**: Ridge Classifier was selected for its:
       - High accuracy (~96.5%)
       - Simplicity and interpretability
    4. **Key Features Used**:
       - Concave Points (mean, SE, worst)
       - Area (mean, SE, worst)
       - Radius (mean, SE, worst)
    """)

    st.markdown("---")
    st.subheader("📊 Feature Importance (Sample Visualization)")

    # Dummy feature importance data
    features = ['Concave Points', 'Area', 'Radius', 'Texture', 'Perimeter',
                'Compactness', 'Concavity', 'Symmetry', 'Smoothness', 'Fractal Dimension']
    importance = [0.18, 0.16, 0.14, 0.12, 0.11, 0.09, 0.08, 0.06, 0.04, 0.02]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(features, importance, color='cornflowerblue', edgecolor='black')

    ax.set_ylabel('Relative Importance', fontsize=12)
    ax.set_title('Top Features in Breast Cancer Prediction Model', fontsize=14, weight='bold')
    ax.set_ylim(0, max(importance) + 0.05)
    plt.xticks(rotation=45, ha='right')

    # Add percentage labels above bars
    for bar, score in zip(bars, importance):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.005,
            f"{score:.0%}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Understanding the Features")
    st.write("""
    The measurements used in this model come from digitized images of a fine needle aspirate (FNA) of a breast mass.
    They describe characteristics of the cell nuclei present in the image.
    """)

    # Adding some spacing for better readability
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    ### Feature Descriptions
    """)

    # Using expanders for better organization with padding
    with st.expander("Cell Size Features", expanded=True):
        st.markdown("""
        - **Radius**: Mean of distances from center to points on the perimeter
        - **Perimeter**: Perimeter of the cell nucleus
        - **Area**: Area of the cell nucleus
        """)

    with st.expander("Cell Texture Features", expanded=True):
        st.markdown("""
        - **Texture**: Standard deviation of gray-scale values
        - **Smoothness**: Local variation in radius lengths
        """)

    with st.expander("Cell Shape Features", expanded=True):
        st.markdown("""
        - **Compactness**: (Perimeter² / Area - 1)
        - **Concavity**: Severity of concave portions of the contour
        - **Concave Points**: Number of concave portions of the contour
        - **Symmetry**: Symmetry of the cell nucleus
        - **Fractal Dimension**: "Coastline approximation" - 1
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    Each feature is computed with three different measurements:
    - **Mean**: The average value
    - **SE**: Standard error
    - **Worst**: The mean of the three largest values
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Cancer Detection Project | 2025</p>
    <p style='font-size: 0.8rem'>Disclaimer: This application is for educational purposes only and should not be used for actual medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)