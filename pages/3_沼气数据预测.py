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

st.title("数据预测")


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Loading up the Regression model we created
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')


# Caching the model for faster loading
@st.cache
# Define the prediction function
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    # Predicting the price of the carat
    if cut == 'Fair':
        cut = 0
    elif cut == 'Good':
        cut = 1
    elif cut == 'Very Good':
        cut = 2
    elif cut == 'Premium':
        cut = 3
    elif cut == 'Ideal':
        cut = 4

    if color == 'J':
        color = 0
    elif color == 'I':
        color = 1
    elif color == 'H':
        color = 2
    elif color == 'G':
        color = 3
    elif color == 'F':
        color = 4
    elif color == 'E':
        color = 5
    elif color == 'D':
        color = 6

    if clarity == 'I1':
        clarity = 0
    elif clarity == 'SI2':
        clarity = 1
    elif clarity == 'SI1':
        clarity = 2
    elif clarity == 'VS2':
        clarity = 3
    elif clarity == 'VS1':
        clarity = 4
    elif clarity == 'VVS2':
        clarity = 5
    elif clarity == 'VVS1':
        clarity = 6
    elif clarity == 'IF':
        clarity = 7

    prediction = model.predict(pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]],
                                            columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y',
                                                     'z']))
    return prediction

from PIL import Image
image = Image.open('./images/bio_gas.jpg')


st.title('预测模块一')
st.image(image)

st.header('请输入沼气相关指标:')

carat = st.number_input('一级发酵罐总体积（立方米）:', min_value=0.1, max_value=10000.0, value=1.0)
temp1 = st.number_input('一级发酵罐-发酵温度（￮C）:', min_value=0.1, max_value=100.0, value=1.0)
pH1=st.number_input('一级发酵罐-pH:', min_value=0.1, max_value=10.0, value=1.0)

two_level_bot=st.number_input('二级发酵罐总体积（立方米）:', min_value=0.1, max_value=10.0, value=1.0)
temp2 = st.number_input('二级发酵罐-发酵温度（￮C）:', min_value=0.1, max_value=100.0, value=1.0)
pH2=st.number_input('二级发酵罐-pH:', min_value=0.1, max_value=10.0, value=1.0)


bio_gas_total_bot=st.number_input('沼液储存罐总体积（立方米）:', min_value=0.1, max_value=10.0, value=1.0)
gas_total_bot=st.number_input('储气罐总体积（立方米）:', min_value=0.1, max_value=10.0, value=1.0)


process_fajiao = st.selectbox('发酵工艺:', ['CSTR', 'USR', 'UASB'])
process_raw = st.selectbox('发酵原料:', ['NawaRo'])


TS_FM = st.number_input('TS (% FM):', min_value=0.1, max_value=100.0, value=1.0)
VS_TS = st.number_input('VS (% of TS):', min_value=0.1, max_value=100.0, value=1.0)
fajiao_TS = st.number_input('发酵液TS (%鲜重):', min_value=0.1, max_value=100.0, value=1.0)
fajiao_VS = st.number_input('发酵液VS (%TS):', min_value=0.1, max_value=100.0, value=1.0)
ORGAN_KG = st.number_input('有机负荷[kg/(m³d)]:', min_value=0.1, max_value=100.0, value=1.0)
WATER_POWER_TIME = st.number_input('水力停留时间（天）:', min_value=0.1, max_value=100.0, value=1.0)
PH3 = st.number_input('pH:', min_value=0.1, max_value=100.0, value=1.0)
FAJIAO_ANDAN = st.number_input('发酵液氨氮含量（g/kg）:', min_value=0.1, max_value=100.0, value=1.0)
FAJIAO_ZONGDAN = st.number_input('发酵液总氮含量（g/kg）:', min_value=0.1, max_value=100.0, value=1.0)
FAJIAO_YOUJISUAN = st.number_input('发酵液有机酸含量（mg/l）:', min_value=0.1, max_value=100.0, value=1.0)
FOS_TAC = st.number_input('FOS/TAC	CHP (kW):', min_value=0.1, max_value=100.0, value=1.0)
WORKING_TIME = st.number_input('工作时间（小时）:', min_value=0.1, max_value=100.0, value=1.0)
BIO_GAS_PRODUCE = st.number_input('沼气产量（m3）:', min_value=0.1, max_value=100.0, value=1.0)
JIAWAN_PRODUCE= st.number_input('甲烷含量[vol%]:', min_value=0.1, max_value=100.0, value=1.0)
z = st.number_input('产电量（度）:', min_value=0.1, max_value=100.0, value=1.0)
z = st.number_input('产热量（kWh):', min_value=0.1, max_value=100.0, value=1.0)

if st.button('产电量预测:'):
    price = predict(carat, TS_FM)
    st.success(f'产电量（度）: ${price[0]:.2f}')

if st.button('产热量预测:'):
    price = predict(carat, TS_FM)
    st.success(f'产热量（kWh): ${price[0]:.2f}')