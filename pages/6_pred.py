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

# Load data function
def load_data(file):
    data = pd.read_csv(file)
    return data

# Create XGBoost model function
def create_model(train_data, target, **params):
    model = xgb.XGBRegressor(**params)
    model.fit(train_data, target)
    return model

# Prediction function
def predict(model, test_data):
    predictions = model.predict(test_data)
    return predictions

# Streamlit app

st.title("XGBoost Model Builder")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Display uploaded data
    st.subheader("Data")
    st.write(data)

    # Select target column
    target_col = st.selectbox("Select target column", data.columns)

    # Select model parameters
    n_estimators = st.slider("Number of estimators", 100, 1000, 500)
    max_depth = st.slider("Max depth", 1, 10, 5)

    # Train model
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    model_params = {"n_estimators": n_estimators, "max_depth": max_depth}
    model = create_model(X, y, **model_params)

    # Predict
    st.subheader("Predictions")
    predictions = predict(model, X)
    st.write(predictions)

    # Download predictions
    output = pd.DataFrame({"Predictions": predictions})
    output_csv = output.to_csv(index=False)
    href = f'<a href="data:file/csv;base64,{b64encode(output_csv.encode()).decode()}" download="predictions.csv">Download Predictions CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
