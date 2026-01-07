import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_model():
    with open("model (1).pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("Random Forest Classifier")

st.write(f"Model expects {model.n_features_in_} features")

# ---- INPUTS (6 FEATURES) ----
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)
f5 = st.number_input("Feature 5", value=0.0)
f6 = st.number_input("Feature 6", value=0.0)

# Combine inputs in EXACT order
input_data = np.array([[f1, f2, f3, f4, f5, f6]])

# ---- PREDICT ----
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)

        st.success(f"Prediction: {prediction}")
        st.write("Prediction probabilities:", probability)
    except Exception as e:
         st.error(f"Error during prediction: {e}")