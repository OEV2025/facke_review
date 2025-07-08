import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

@st.cache_data
def load_model():
    data = pd.read_csv("deceptive-opinion.csv")

    # Verifică coloanele
    st.write("Coloane detectate:", data.columns.tolist())

    # Folosim coloana corectă pentru text și etichetă
    X = data["text"]
    y = data["deceptive"]

    # Construim un model simplu TF-IDF + Logistic Regression
    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X, y)
    return model

st.title("🕵️‍♂️ Detector Recenzii False")

user_input = st.text_area("Scrie o recenzie turistică:")

if user_input:
    model = load_model()
    prediction = model.predict([user_input])[0]

    if prediction == 1:
        st.error("🔴 Această recenzie pare FALSĂ.")
    else:
        st.success("🟢 Această recenzie pare AUTENTICĂ.")
