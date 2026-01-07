import streamlit as st
import pickle
import pandas as pd
import numpy as np

MODEL_PATH = "/mnt/data/model (1).pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🏡 House Price Prediction App")

# Extract feature names from pipeline
preprocess = model.named_steps["preprocess"]

num_features = preprocess.transformers_[0][2]
cat_features = preprocess.transformers_[1][2]

st.subheader("Enter House Details")

input_data = {}

# Numeric Inputs
st.markdown("### 🔢 Numeric Features")
for col in num_features:
    input_data[col] = st.number_input(
        col.replace("_", " ").title(),
        value=0.0,
        format="%.2f"
    )

# Categorical Inputs
st.markdown("### 🔠 Categorical Features")
for col in cat_features:
    input_data[col] = st.selectbox(
        col.replace("_", " ").title(),
        ["Yes", "No", "Good", "Average", "Poor", "Type1", "Type2"]
    )

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Predict Button
if st.button("Predict Price"):
    try:
        prediction = model.predict(df)[0]
        st.success(f"🏠 Estimated House Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
