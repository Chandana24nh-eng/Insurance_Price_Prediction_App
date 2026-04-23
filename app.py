# create environment for windows
# python -m venv env
# activate environment
# myenv\Scripts\activate
# pip install streamlit scikit-learn pandas seaborn numpy
import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# load model
model=pickle.load(open('model.pkl','rb'))

# initalize scaler used in insurance price prediction model
scaler=StandardScaler()

# to give title
st.title("Insurance Price Prediction APP")

# import input variables
age=st.number_input("Age",min_value=1,max_value=100,value=20)
gender=st.selectbox('Gender',['male', 'female'])
bmi=st.number_input("BMI",min_value=10.0,max_value=80.0,value=30.0)
smoker=st.selectbox('Smoker',['yes','no'])
children=st.number_input("Children",min_value=0,max_value=10,value=1)
region=st.selectbox('Region',['southwest','southeast','northwest','northeast'])

# encode

# smoker: yes, no
Smoker = 1 if smoker == 'yes' else 0

# gender: male, female
sex_female = 1 if gender == 'female' else 0
sex_male = 1 if gender == 'male' else 0

# region
region_dict = {'southeast': 3, 'northeast': 2, 'northwest': 1, 'southwest': 0}
Region = region_dict[region]

# create dataframe that includes all the variables -->should be same as in the data of the insurance price prediction model,[54]th row
input_features = pd.DataFrame({
    'age':[age],
    'bmi':[bmi],
    'children':[children],
    'Smoker':[Smoker],
    'sex_female':[sex_female],
    'sex_male':[sex_male],
    'Region':[Region]
})

# apply scaling
#input_features[['age','bmi']]=scaler.fit_transform(input_features[['age','bmi']])

# make predictions
if st.button('Predict'):
  predictions = model.predict(input_features)
  output = round(predictions[0], 2)
  st.success(f'Price Prediction: ${output:,.2f}')
  
