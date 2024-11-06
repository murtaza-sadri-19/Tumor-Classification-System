# **Breast Cancer Classification Model**

### Overview
This repository contains a breast cancer classification model implemented in Python using scikit-learn.The model is trained on the [Breast Cancer Wisconsin (Diagnostic) dataset] (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and predicts whether a tumor is malignant or benign.

### Model
The model used in this project is a logistic regression classifier, which is trained on the training set and evaluated on the testing set. The model's performance is measured using accuracy and F1-score metrics.

### Features
The dataset contains 32 features, which are divided into two categories: discrete and continuous. The discrete features are:

**diagnosis**: the diagnosis of the tumor (malignant or benign)
The continuous features are:

* radius_mean
* texture_mean
* perimeter_mean
* area_mean
* smoothness_mean
* compactness_mean
* concavity_mean
* concave points_mean
* symmetry_mean
* fractal dimension_mean
* radius_se
* texture_se
* perimeter_se
* area_se
* smoothness_se
* compactness_se
* concavity_se
* concave points_se
* symmetry_se
* fractal dimension_se
* radius_worst
* texture_worst
* perimeter_worst
* area_worst
* smoothness_worst
* compactness_worst
* concavity_worst
* concave points_worst
* symmetry_worst
* fractal dimension_worst

### Usage
To use the model, simply run the breast_cancer_model.py script and provide the input data as a tuple of 30 values, representing the 30 continuous features of the dataset. The model will predict whether the tumor is malignant or benign.

### Model Performance
The model's performance is evaluated using __*accuracy*__ and __*F1-score*__ metrics. The accuracy of the model is 0.96.5, and the F1-score is 0.951.

### Model Deployment
The model is deployed using the pickle library, which allows us to save the trained model to a file and load it later for prediction.

I hope this README file helps! Let me know if you have any questions or need further clarification. :) 
