import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguins Prediction app

This app predict the **Palmer Penguin** species!
  
""")

st.sidebar.header('User input feature')

uploaded_file = st.sidebar.file_uploader('Upload your data here',type= ['csv'])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe', 'Dream', "Torgersen"))
        sex = st.sidebar.selectbox('Sex',('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32,60,44)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13,21,17)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 170, 230, 200)
        body_mass_g = st.sidebar.slider(' Body Mass (g)', 2700, 6300, 4200)
        data = {
            'island':island,
            'bill_length_mm' : bill_length_mm,
            'bill_depth_mm':bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex' : sex
            }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns= ['species'])
df = pd.concat([input_df, penguins], axis = 0)

encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix= col)
    df = pd.concat([df,dummy], axis = 1)
    del df[col]

st.subheader('User Input features')

if uploaded_file is not None:
    df = df.drop(columns= ['species'])
    st.write(df)
else:
    st.write(' Awaiting CSV file to be upload. Currently using example input parameters.')
    df = df[:1]
    st.write(df)


load_clf = pickle.load(open('model.pkl','rb'))
prediction = load_clf.predict(df)
prediction_prob = load_clf.predict_proba(df)

st.subheader('Prediction')
species = np.array(['Adelie', 'Chinstrap', "Gentoo"])
st.write(species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)
