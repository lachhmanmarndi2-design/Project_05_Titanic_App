
import streamlit as st
import pandas as pd
import joblib

FEATURES = ["pclass", "sex", "age", "fare"]

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.title("Titanic Survival Predictor")
st.write("Enter passenger details below.")

pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Sex", ["female", "male"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare", 0, 500, 30)

if st.button("Predict"):
    try:
        model = load_model()
        sex_value = 1 if sex == "female" else 0

        data = pd.DataFrame(
            [[pclass, sex_value, age, fare]],
            columns=FEATURES
        )

        probability = model.predict_proba(data)[0][1]

        st.metric(
            "Estimated Survival Chance",
            f"{probability:.1%}"
        )

    except Exception:
        st.error("Please check the model.joblib file.")

st.caption("Educational project using historical Titanic data.")
