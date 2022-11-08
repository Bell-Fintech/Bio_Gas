import xgboost as xgb
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


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

st.title("经济效益预测")


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Loading up the Regression model we created
model = xgb.XGBRegressor()
model.load_model('./models/economy_amount_of_electricity_fed_into_the_grid_predict.json')


from PIL import Image
image = Image.open('./images/bio_gas.jpg')


st.title('预测模块三')
st.image(image)

st.header('请输入发电相关经济指标:')
maximum_rated_electrical_power = st.number_input('maximum_rated_electrical_power', 1, 99999, 1)
substrate_costs = st.number_input('substrate_costs', 1, 99999, 1)
personnel_costs = st.number_input('personnel_costs', 1, 99999, 1)
maintenance_costs = st.number_input('maintenance_costs', 1, 99999, 1)
depreciation = st.number_input('depreciation', 1, 99999, 1)
Other_operating_costs = st.number_input('Other_operating_costs',1, 99999, 1)

features = [[maximum_rated_electrical_power,substrate_costs,personnel_costs,maintenance_costs,depreciation,Other_operating_costs]]


# Loading up the Regression model we created
# = pickle.load(open('./models/economy_amount_of_electricity_fed_into_the_grid_predict.pkl', 'rb'))

if st.button('amount of electricity fed into the grid（kWh）:'):
    prediction = model.predict(features)

    st.success(f'amount of electricity fed into the grid（kWh）: {prediction[0]:.4f}')

