import streamlit as st
import numpy as np
import pandas as pd
import joblib
model = joblib.load('xgb_model.pkl')
pipeline = joblib.load('pipeline.pkl')

cat_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
num_cols = ['Age', 'Work Pressure', 'Job Satisfaction', 'Work Hours', 'Financial Stress']


st.title("Mental Health Prediction App")
st.caption("Please take in mind that this app was developed solely for interest. It do not guarantee anything.")
st.sidebar.header("User Input Features")


input_data = {}


for col in cat_cols:
    unique_values = ['Female', 'Male'] if col == "Gender" else \
                    ['7-8 hours', '5-6 hours', 'More than 8 hours', 'Less than 5 hours'] if col == "Sleep Duration" else \
                    ['Moderate', 'Unhealthy', 'Healthy'] if col == "Dietary Habits" else \
                    ['No', 'Yes']  # For other binary columns
    input_data[col] = st.sidebar.selectbox(col, unique_values)


for col in num_cols:
    min_val, max_val = (18, 60) if col == "Age" else (0, 12) if col == "Work Hours" else (1, 5)
    input_data[col] = st.sidebar.slider(col, min_val, max_val, int((min_val + max_val) / 2))


if st.sidebar.button("Submit"):

    input_df = pd.DataFrame([input_data])
    X = pipeline.transform(input_df)
    prediction = model.predict(X)
    y_pred = int(prediction[0])
    if y_pred == 0:
        st.title("Glad to hear!")
        st.text("It is unlikely that you have depression. But it is still better not to forget to take care of yourself!")
    else:
        st.title("Bad news")
        st.text("According to statistics, it is possible that you have a depression. Please check up your mental health!")
    st.write(f"The prediction is: **{prediction[0]}**")