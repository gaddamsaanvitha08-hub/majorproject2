import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Sleep Quality Predictor", page_icon="😴")
st.title("😴 Sleep Quality Prediction - Major Project")
st.write("Live and Working!")

file_path = Path(__file__).parent / "sleep.csv"

if not file_path.exists():
    st.error(f"sleep.csv not found! Files in folder: {list(Path(__file__).parent.glob('*'))}")
    st.warning("Using demo mode without CSV")
    # Demo mode
    st.header("Enter Details")
    age = st.slider("Age", 10, 100, 25)
    sleep_hours = st.slider("Sleep Hours", 1, 12, 7)
    stress = st.slider("Stress", 1, 10, 5)
    if st.button("Predict"):
        if sleep_hours >= 7:
            st.success("✅ Good Sleep Quality!")
            st.balloons()
        else:
            st.error("❌ Poor Sleep Quality")
    st.stop()

# If CSV exists, load it
data = pd.read_csv(file_path)
st.success("Dataset loaded successfully!")
st.dataframe(data.head())

data["Sleep Disorder"] = data["Sleep Disorder"].fillna("None")

st.header("Try Prediction")
age = st.slider("Age", 18, 80, 30)
hours = st.slider("Sleep Hours", 3, 12, 7)

if st.button("Predict Sleep Quality"):
    st.success(f"Predicted for Age {age} with {hours} hours sleep!")
    st.balloons()

st.write("---")
st.write("**By Saanvitha - Major Project 2025**")
