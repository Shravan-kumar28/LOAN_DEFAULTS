# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:59:03 2025

@author: shrav
"""

import streamlit as st
import pandas as pd
import joblib
import pickle

# Streamlit page configuration
st.set_page_config(page_title="Decision Tree Classifier", page_icon="📊")

# Title and description
st.title("Decision Tree Classifier with Hyperparameter Tuning")
st.markdown("""
This app uses a Decision Tree Classifier on a dataset to predict the 'Sales' category (Low or High) 
and visualizes the performance of the model through cross-validation and accuracy metrics.
""")

# File upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset from the uploaded file
    df = pd.read_csv(r"E:\360DigiTmg\360downloads\Problem Statement\ML\Decision tree\Data Set DT\ClothCompany_Data.csv")
    
    # Ensure you have the correct paths for your model and preprocessing objects
    try:
        # Load pre-trained model and preprocessing objects
        model = pickle.load(open('Dtree_best.pkl', 'rb'))
        Scale_one = joblib.load('Scalar_Onehot')
        Winsor = joblib.load('Winsor')

        # Preprocess data: apply scaling and outlier transformation
        CleanData = pd.DataFrame(Scale_one.transform(df), columns=Scale_one.get_feature_names_out())
        CleanData[['num__CompPrice', 'num__Price']] = Winsor.transform(CleanData[['num__CompPrice', 'num__Price']])

        # Make predictions
        prediction = pd.DataFrame(model.predict(CleanData), columns=['Sales Prediction'])

        # Concatenate predictions with original dataset
        final_data = pd.concat([df, prediction], axis=1)

        # Display the final data with predictions
        st.subheader("Predicted Results")
        st.write(final_data)

    except Exception as e:
        st.error(f"Error loading the model or preprocessing objects: {e}")

else:
    st.warning("Please upload a CSV file to get started!")
