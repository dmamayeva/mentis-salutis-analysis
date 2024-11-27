# mentis-salutis-analysis
# Description
This project aims to predict the likelihood of depression in individuals by analyzing the relationship between mental health and various demographic, lifestyle, and work-related factors. The dataset includes variables such as gender, age, work pressure, job satisfaction, sleep duration, and financial stress, with the goal of identifying key risk factors and providing insights into the impact of work-life balance and lifestyle on mental well-being."
# Demo app
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dmamayeva-mentis-salutis-analysis-streamlit-app-qyfxju.streamlit.app)
# Dataset info
The dataset is available at [Kaggle]([https://website-name.com](https://www.kaggle.com/datasets/ikynahidwin/depression-professional-dataset))
# How to run
```
docker build -t mentis-salutis .
```
```
docker run -it -p 9696:9696 mentis-salutis:latest
```
As request:
```
import requests
url = "http://0.0.0.0:9696/predict"

example_request = {"Gender":"Female",
"Age":37,
"Work Pressure":2.0,
"Job Satisfaction":4.0,
"Sleep Duration":"7-8 hours",
"Dietary Habits":"Moderate",
"Have you ever had suicidal thoughts ?":"No",
"Work Hours":6,
"Financial Stress":2,
"Family History of Mental Illness":"No"}

requests.post(url, json=example_request).json()
```
