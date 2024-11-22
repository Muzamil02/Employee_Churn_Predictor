#Importing Necessary Libraries
import streamlit as st
import pandas as pd
from joblib import load
import dill

# Loading Pretrained model
with open('pipeline.pkl','rb') as file:
	model = dill.load(file)

my_feature_dict = load('my_feature_dict.pkl')

#Functions to predict
def predict_churn(data):
	prediction = model.predict(data)
	return prediction

st.title('EMPLOYEE CHURN PREDICTOR')
st.subheader('Created By MUZAMIL HUSSAIN')

#Categorical features
st.subheader('Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals={}
for i, col in enumerate(categorical_input.get('Column Name').values()):
	categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i])

#Numerical features
numerical_input = my_feature_dict.get('NUMERICAL')

st.subheader('Numerical Features')
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals={}
for col in numerical_input.get('Column Name', []):
    if col == 'JoiningYear':
        numerical_input_vals[col] = st.number_input(
            col, min_value=2012, max_value=2018, step=1, format='%d'
        )
    elif col == 'Age':
        numerical_input_vals[col] = st.number_input(
            col, min_value=22, max_value=41, step=1, format='%d'
        )
    elif col == 'PaymentTier':
        numerical_input_vals[col] = st.number_input(
            col, min_value=1, max_value=3, step=1, format='%d'
        )
    elif col == 'ExperienceInCurrentDomain':
        numerical_input_vals[col] = st.number_input(
            col, min_value=0, max_value=7, step=1, format='%d'
        )
    else:
        numerical_input_vals[col] = st.number_input(col)

#Combining both numerical and categorical vals
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))

input_data = pd.DataFrame.from_dict(input_data,orient='index').T

#Churn Prediction
if st.button("Predict"):
    prediction = predict_churn(input_data)[0]
    translate_dict = {"Yes" : "Churn", "No" : "Stay"}
    prediction_translate = translate_dict.get(prediction)
    st.subheader(f'The Prediction is **{prediction}**, The Employee is expected to **{prediction_translate}**.')

    st.subheader('THANKYOU')