import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Sample Dataset (you can replace later with CSV)
# -------------------------------
data = {
    'fever': [1, 1, 0, 0, 1, 0],
    'cough': [1, 0, 1, 0, 1, 0],
    'headache': [1, 1, 0, 0, 1, 0],
    'fatigue': [1, 1, 0, 0, 1, 0],
    'disease': ['Flu', 'Flu', 'Cold', 'Healthy', 'Flu', 'Healthy']
}

df = pd.DataFrame(data)

# -------------------------------
# Encode target
# -------------------------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['disease'] = le.fit_transform(df['disease'])

# -------------------------------
# Train Model
# -------------------------------
X = df.drop('disease', axis=1)
y = df['disease']

model = RandomForestClassifier()
model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 Intelligent Medical Diagnosis Assistant")

st.write("Select symptoms:")

fever = st.selectbox("Fever", [0, 1])
cough = st.selectbox("Cough", [0, 1])
headache = st.selectbox("Headache", [0, 1])
fatigue = st.selectbox("Fatigue", [0, 1])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Disease"):
    input_data = [[fever, cough, headache, fatigue]]
    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)

    st.success(f"Predicted Disease: {result[0]}")