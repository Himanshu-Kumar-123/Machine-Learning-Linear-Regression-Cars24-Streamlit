import streamlit as st
import pandas as pd
import pickle

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fdf6e3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load saved objects ---
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("make_map.pkl", "rb") as f:
    make_map = pickle.load(f)

with open("model_map.pkl", "rb") as f:
    model_map = pickle.load(f)

# --- Page Configuration ---
st.set_page_config(page_title="Car Resale Price Predictor", layout="centered")

# --- App Title and Image ---
st.title("🚗 Car Resale Price Predictor")
st.image("car.png", caption="Your Dream Ride Awaits 🚘", use_container_width=True)

st.markdown("---")

# --- User Input ---
st.subheader("🔧 Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Select Make", list(make_map.keys()))
    age = st.number_input("Car Age (in years)", min_value=0, max_value=30, step=1)
    km_driven = st.slider("Kilometers Driven", min_value=0, max_value=300000, step=1000, value=50000)

with col2:
    df_raw = pd.read_csv("cars24-car-price.csv")
    filtered_models = df_raw[df_raw['make'] == make]['model'].unique()
    model_choice = st.selectbox("Select Model", sorted(filtered_models))
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# --- Convert inputs ---
make_val = make_map[make]
model_val = model_map[model_choice]
petrol_val = 1 if fuel == "Petrol" else 0
manual_val = 1 if transmission == "Manual" else 0

# --- Prepare input DataFrame ---
input_df = pd.DataFrame([[
    km_driven, age, make_val, model_val, petrol_val, manual_val
]], columns=['km_driven', 'age', 'make', 'model', 'Petrol', 'Manual'])

st.markdown("---")

# --- Predict Button ---
centered_button = """
    <div style="display: flex; justify-content: center;">
        <button style="background-color:#4CAF50;color:white;padding:10px 24px;font-size:16px;border:none;border-radius:5px;">
            Predict Resale Price
        </button>
    </div>
"""
if st.button("Predict Resale Price"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"💰 Estimated Resale Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
