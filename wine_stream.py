import streamlit as st
import pandas as pd
import pickle

st.title("Model to predict quality of Wine")
st.write("we'll use chemical characteristics to predict the quality of one bottle of wine")
st.write("The classification has been divided in 3 categories: low, medium and high")
st.write("---")
st.header("Enter the wine characteristics:")

fixed_acidity = st.text_input("Fixed acidity:")
volatile_acidity = st.text_input("Volatile acidity:")
citric_acid = st.text_input("Citric acid:")
residual_sugar = st.text_input("Residual sugar:")
chlorides = st.text_input("Chlorides:")
density = st.text_input("Density:")
sulphates = st.text_input("Sulphates:")
alcohol = st.text_input("Alcohol:")
data={"fixed_acidity": [float(fixed_acidity)],
      "volatile_acidity": [float(volatile_acidity)],
      "citric_acid": [float(citric_acid)],
      "residual_sugar": [float(residual_sugar)],
      "chlorides": [float(chlorides)],
      "density": [float(density)],
      "sulphates": [float(sulphates)],
      "alcohol": [float(alcohol)]
     }
df = pd.DataFrame(data)
st.write(df)
loaded_model=pickle.load(open("model_wine.p","rb"))
if st.button("Predict"):
    prediction = loaded_model.predict(df)
    if prediction[0]==0:
        st.write("Your wine has low quality")
    elif prediction[0]==1:
        st.write("Your wine has medium quality")
    else:
        st.write("Your wine has high quality")

