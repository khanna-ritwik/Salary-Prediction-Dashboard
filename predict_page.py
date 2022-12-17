import streamlit as st
import pickle
import numpy as np
from PIL import Image


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    col1, col2, col3 = st.columns([1, 30, 1])

    with col1:
        st.write("")

    with col2:
        st.title("Predict Software Engineer Salary")

        st.write("""### We need some information to predict the salary""")

        countries = (
            "United States of America",
            "Germany",
            "United Kingdom of Great Britain and Northern Ireland",
            "India",
            "Canada",
            "France",
            "Brazil",
            "Spain",
            "Netherlands",
            "Australia",
            "Italy",
            "Poland",
            "Sweden",
            "Russian Federation",
            "Switzerland"
        )

        education = (
            'Less than a Bachelors',
            'Bachelor’s degree',
            'Master’s degree',
            'Post grad'
        )
        image = Image.open('predict_logo.jpg')
        new_image = image.resize((200, 100))
        st.image(new_image)
        country = st.selectbox("Country", countries)
        education = st.selectbox("Education Level", education)

        experience = st.slider("Years of Experience", 0, 50, 3)

        ok = st.button('Predict Salary')
        if ok:
            X = np.array([[country, education, experience]])
            X[:, 0] = le_country.transform(X[:, 0])
            X[:, 1] = le_education.transform(X[:, 1])
            X = X.astype(float)

            salary = regressor.predict(X)
            st.subheader(f"The estimated salary is ${salary[0]:.2f}")

    with col3:
        st.write("")
