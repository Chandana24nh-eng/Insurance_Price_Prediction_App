import os
import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------- UI CONFIG ---------------- #
st.set_page_config(page_title="Insurance Predictor", page_icon="💰", layout="centered")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    text-align:center;
    font-size:32px;
    font-weight:bold;
    color:#00c6ff;
}
.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
.card {
    background:#1c1f26;
    padding:20px;
    border-radius:12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
}
.result {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:white;
    font-size:24px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)



model_path = os.path.join(os.path.dirname(__file__), 'gb_model.pkl')

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open('gb_model.pkl','rb'))

# scaler
scaler = StandardScaler()

# ---------------- HEADER ---------------- #
st.markdown('<div class="title">💰 Insurance Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter details to estimate cost</div>', unsafe_allow_html=True)

# ---------------- INPUT CARD ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=1, max_value=100, value=20)
    bmi = st.number_input('BMI', min_value=10.0, max_value=80.0, value=30.0)
    children = st.number_input('Number of children', min_value=0, max_value=10, value=2)

with col2:
    gender = st.selectbox('Gender', ('male','female'))
    smoker = st.selectbox('Smoker', ('yes','no'))
    region = st.selectbox('Region', ('southwest','southeast','northwest','northeast'))

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- ENCODING ---------------- #
Smoker = 1 if smoker == 'yes' else 0
sex_female = 1 if gender == 'female' else 0
sex_male = 1 if gender == 'male' else 0

region_dict = {'southeast':3,'northeast':2,'northwest':1,'southwest':0}
Region = region_dict[region]

# ---------------- DATAFRAME ---------------- #
input_features = pd.DataFrame({
    'age':[age],
    'bmi':[bmi],
    'children':[children],
    'Smoker':[Smoker],
    'sex_female':[sex_female],
    'sex_male':[sex_male],
    'Region':[Region]
})

input_features[['age','bmi']] = scaler.fit_transform(input_features[['age','bmi']])

# ---------------- BUTTON ---------------- #
if st.button('🚀 Predict Price', use_container_width=True):

    # feature alignment fix (kept minimal)
    try:
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in input_features.columns:
                input_features[col] = 0
        input_features = input_features[model_features]
    except:
        pass

    predictions = model.predict(input_features)
    output = round(np.exp(predictions[0]), 2)

    st.markdown("<br>", unsafe_allow_html=True)

    # RESULT BOX
    st.markdown(f'<div class="result">Estimated Cost: ${output:,.2f}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # STATUS
    col1, col2 = st.columns(2)

    with col1:
        if Smoker:
            st.error("🚬 Smoker")
        else:
            st.success("✅ Non-Smoker")

    with col2:
        st.info(f"👤 {gender.capitalize()}")

# ---------------- FOOTER ---------------- #
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<center style='color:gray;'>🚀 Built with Streamlit</center>", unsafe_allow_html=True)
