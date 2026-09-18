import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

st.title("Electric Bill Prediction")

st.write("Enter AC Units and Fan Units to predict the Electric Bill.")

# Load the trained model
model = joblib.load("electric_bill_prediction_model.pkl")

# Create polynomial transformer
poly = PolynomialFeatures(degree=2)

# AC Units
ac_units = st.number_input(
    "Enter AC Units",
    min_value=0,
    step=1
)

# Fan Units
fan_units = st.number_input(
    "Enter Fan Units",
    min_value=0,
    step=1
)

# Prediction
if st.button("Predict Electric Bill"):

    # Check if both are zero
    if ac_units == 0 and fan_units == 0:
        st.error("AC Units and Fan Units cannot both be zero.")

    else:
        # Create input data
        input_data = np.array([[ac_units, fan_units]])

        # Polynomial transformation
        input_data_poly = poly.fit_transform(input_data)

        # Prediction
        prediction = model.predict(input_data_poly)

        # Display result
        st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")
