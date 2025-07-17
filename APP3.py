import streamlit as st
import pickle



a = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
     'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
     'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
     'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(a)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

N = st.text_input("Enter Nitrogen value:")
P = st.text_input("Enter Phosphorus value:")
K = st.text_input("Enter Potassium value:")
temperature = st.text_input("Enter temperature value:")
humidity = st.text_input("Enter humidity value:")
ph = st.text_input("Enter pH value:")
rainfall = st.text_input("Enter the rainfall:")

if st.button("Submit"):
    data = [[N, P, K, temperature, humidity, ph, rainfall]]
    pred = model.predict(data)
    predict = le.inverse_transform(pred)
    
    st.write(f"Prediction: {predict[0]}")
