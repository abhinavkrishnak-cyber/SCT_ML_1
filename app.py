import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("housing.csv")

# Train model
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.title("🏠 House Price Prediction Dashboard")

area = st.slider("Area", 500, 15000, 5000)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)

if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Estimated House Price: ₹{prediction[0]:,.2f}")
