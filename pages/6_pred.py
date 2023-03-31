import xgboost as xgb
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.sidebar.title("沼气学术会议")
st.sidebar.info(
    """
    中国沼气学会学术年会:     
    <http://www.biogaschina.com.cn>         
    能源系统与电气工程会议:       
    <https://www.ncsti.gov.cn>
    """
)

st.sidebar.title("友情链接")
st.sidebar.info(
    """
    国际沼气网: <http://www.biogasintel.com>       
    中国农业农村部: <http://www.moa.gov.cn>          
    香港可再生能源网: <https://re.emsd.gov.hk>

    """
)

st.title("XGBoost沼气产量预测模型训练")

import streamlit as st
import pandas as pd
import xgboost as xgb

def load_data(file):
    data = pd.read_csv(file)
    return data

def create_model(train_data, target, **params):
    model = xgb.XGBRegressor(**params)
    model.fit(train_data, target)
    return model

def predict(model, test_data):
    predictions = model.predict(test_data)
    return predictions

def download_csv(data, filename):
    csvfile = data.to_csv(index=False)
    return csvfile.encode()

st.set_page_config(page_title="My Streamlit App", page_layout="wide")

st.title('XGBoost Model Builder')

# Upload data
st.sidebar.subheader('Upload data')
file = st.sidebar.file_uploader('Choose a CSV file', type='csv')

if file is not None:
    data = load_data(file)

    # Display data
    st.subheader('Data preview')
    st.write(data.head())

    # Select target column
    st.sidebar.subheader('Select target column')
    target_col = st.sidebar.selectbox('Target column', data.columns)

    # Select features columns
    st.sidebar.subheader('Select features columns')
    features_cols = st.sidebar.multiselect('Features columns', data.columns)

    # Select model parameters
    st.sidebar.subheader('Select model parameters')
    params = {}
    params['n_estimators'] = st.sidebar.slider('Number of estimators', 100, 1000, 500)
    params['max_depth'] = st.sidebar.slider('Max depth', 1, 10, 5)

    # Train model
    X = data[features_cols]
    y = data[target_col]
    model = create_model(X, y, **params)

    # Prediction
    st.subheader('Prediction')
    inputs = {}
    for feature_col in features_cols:
        inputs[feature_col] = st.number_input(feature_col, min_value=0.0, step=0.01)
    inputs_df = pd.DataFrame(inputs, index=[0])
    prediction = predict(model, inputs_df)
    st.write(f'The prediction is {prediction[0]:,.2f}.')

    # Download predictions
    if st.button('Download predictions'):
        csvfile = download_csv(inputs_df, 'predictions')
        st.download_button(label="Download predictions CSV", data=csvfile, file_name='predictions.csv', mime='text/csv')
