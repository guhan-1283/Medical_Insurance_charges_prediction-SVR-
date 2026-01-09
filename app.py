import streamlit as st
import numpy as np
import pandas as pd


# Page config

st.set_page_config(
    page_title="Medical Insurance Charges Prediction",
    layout="centered"
)

st.title("Medical Insurance Charges Prediction")
st.write("Predict medical insurance charges using an SVR model")

st.divider()


# Train model INSIDE app 

@st.cache_resource
def train_model():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR

    df = pd.read_csv("medical_insurance.csv")

    X = df.drop("charges", axis=1)
    y = np.log1p(df["charges"])

    num_cols = X.select_dtypes(exclude="object").columns
    cat_cols = X.select_dtypes(include="object").columns

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("svr", SVR(kernel="rbf", C=100, gamma=0.05, epsilon=0.1)),
        ]
    )

    model.fit(X, y)
    return model


model = train_model()  # 👈 model is created here


# User Inputs 
age = st.slider("Age", 18, 70, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 45.0, 30.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Prediction

if st.button("Predict Insurance Charges"):
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

    st.success(f"Estimated Insurance Charges: ₹{prediction:,.2f}")
