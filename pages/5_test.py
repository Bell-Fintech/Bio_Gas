import streamlit as st
import pandas as pd
import numpy as np
import pickle
import socket
socket.setdefaulttimeout(500)
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.header('User Input Features')

st.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        maximum_rated_electrical_power = st.slider('maximum_rated_electrical_power',32.1,59.6,43.9)
        substrate_costs = st.slider('substrate_costs',172.0,231.0,201.0)
        personnel_costs = st.slider('personnel_costs', 32.1,59.6,43.9)
        maintenance_costs = st.slider('maintenance_costs', 13.1,21.5,17.2)
        depreciation = st.slider('depreciation', 172.0,231.0,201.0)
        Other_operating_costs = st.slider('Other_operating_costs', 1000.0,6300000.0,4207.0)
        data = {'maximum_rated_electrical_power': maximum_rated_electrical_power,
                'substrate_costs': substrate_costs,
                'personnel_costs': personnel_costs,
                'maintenance_costs': maintenance_costs,
                'depreciation': depreciation,
                'Other_operating_costs': Other_operating_costs}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'], axis=1)
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('./models/economy_amount_of_electricity_fed_into_the_grid_predict.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df,validate_features=False)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
