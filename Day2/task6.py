import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
st.title("Student Score Prediction System")
df = pd.read_csv(r"C:\Users\Harsini\Downloads\data.csv")
X = df[["Hours"]]
y = df["Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
st.write(f"### Model Accuracy (R² Score): {accuracy:.4f}")
hours = st.number_input("Enter Study Hours",min_value=0.0,max_value=24.0,value=1.0,step=0.5,)
if st.button("Predict Score"):
    input_data = pd.DataFrame({"Hours": [hours]})
    prediction = model.predict(input_data)
    st.success(f"Predicted Score: {prediction[0]:.2f}")