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


from PIL import Image
image = Image.open('./images/bio_gas.jpg')


st.title('预测模块')
st.image(image)

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import os

# 上传数据集
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file,encoding='utf-8')

    # 特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 超参数调整
    max_depth = st.slider("max_depth", min_value=1, max_value=10, value=3, step=1)
    learning_rate = st.slider("learning_rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    n_estimators = st.slider("n_estimators", min_value=1, max_value=1000, value=100, step=1)

    # 模型训练
    model = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
    model.fit(X_scaled, y)

    # 缓存模型
    model_file = "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)


# 预测数据集
predict_file = st.file_uploader("Upload CSV for Prediction", type="csv")
if predict_file is not None:
    predict_data = pd.read_csv(predict_file,encoding='utf-8')

    # 添加空白的标签列
    predict_data["prediction"] = np.nan

    # 归一化
    scaler = MinMaxScaler()
    predict_data_scaled = scaler.fit_transform(predict_data.iloc[:, :-1])

    # 加载模型
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # 预测
        y_pred = model.predict(predict_data_scaled)

        # 输出预测结果
        predict_data["prediction"] = y_pred
        st.write(predict_data)

        # 下载预测结果
        st.download_button("Download Prediction", predict_data.to_csv(index=False), "prediction.csv")
