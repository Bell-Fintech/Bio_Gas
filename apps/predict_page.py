import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('./models/saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("沼气数据预测模块")

    st.write("""### 高精度预测沼气价格""")

    countries = (
        "United States",                          
        "India",                  
        "United Kingdom",         
        "Germany",                
        "Canada",                 
        "France",                 
        "Brazil",                 
        "Australia",              
        "Spain",                  
        "Netherlands",            
        "Poland",                 
        "Russian Federation",     
        "Italy",                  
        "Sweden",                 
        "Israel",                 
        "Turkey",                  
        "Switzerland",             
        "Ukraine",             
        "Mexico",                  
        "Norway",                 
        "Pakistan",                
        "Belgium",                 
        "Austria",                 
        "Romania",                 
        "South Africa",            
        "Czech Republic",          
        "Ireland",                 
        "Denmark",                 
        "Portugal",                
        "Finland",                 
        "Iran",                    
        "Bulgaria",                
        "Argentina",               
        "New Zealand",             
        "Hungary",                 
        "Greece"
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("沼气原料来源", countries)
    education = st.selectbox("沼气原料种类", education)

    expericence = st.slider("沼气原料级别", 0, 50, 3)

    ok = st.button("预测沼气价格")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"沼气价格为： ${salary[0]:.2f}")
        usd = salary
        inr = usd * 7.09
        st.subheader(f"沼气价格为： ￥{inr[0]:.2f}")
    
