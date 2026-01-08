import streamlit as st
import numpy as np
import pandas as pd
import joblib


# load model
model = joblib.load("model.pkl")


# title of app
st.set_page_config(page_title="Medical Insurance charges Prediction",layout="centered")


st.title("Medical Insurance Prediction")

st.write("Predict medical Insurance Charges using SVR model")

st.divider()


# user inputs

age = st.slider("Age",18,70,30)
sex = st.selectbox("sex",["male","female"])
bmi = st.slider("BMI",15.0,45.0,30.0)
children = st.selectbox("No. of Childrens",[0,1,2,3,4,5])
smoker = st.selectbox("Smoker",["yes","no"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)


# prediction 
if st.button("Predd this button yo predict charges"):
    

    

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    log_prediction = model.predict(input_data)
    prediction = np.expm1(log_prediction)[0]



    st.success(f"Enstimate Insurance Charges: Rs.{prediction:,.2f}")

    