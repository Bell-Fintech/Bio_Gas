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
# import streamlit as st
# import pandas as pd
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split


# # 添加上传数据集功能
# uploaded_file = st.file_uploader("上传数据集", type=["csv"])




# # 显示数据集的前5行
# st.write("Data Preview:")
# st.write(data.head())

# # 选择目标变量和特征变量
# target_variable = st.selectbox("Select target variable", data.columns)
# feature_variables = st.multiselect("Select feature variables", data.columns)

# # 拆分数据集为训练集和测试集
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# # 定义XGBoost模型参数
# params = {
#     "max_depth": st.slider("Maximum depth", 1, 10, 3),
#     "learning_rate": st.slider("Learning rate", 0.01, 0.5, 0.1),
#     "n_estimators": st.slider("Number of estimators", 10, 500, 100),
#     "objective": "reg:squarederror"
# }

# # 训练XGBoost模型
# X_train = train_data[feature_variables]
# y_train = train_data[target_variable]
# X_test = test_data[feature_variables]
# y_test = test_data[target_variable]
# model = xgb.XGBRegressor(**params)
# model.fit(X_train, y_train)

# # 在测试集上评估模型性能
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# st.write("Mean squared error:", mse)

# # 显示特征重要性
# st.write("Feature importances:")
# importance_df = pd.DataFrame({
#     "Feature": feature_variables,
#     "Importance": model.feature_importances_
# }).sort_values(by="Importance", ascending=False)
# st.bar_chart(importance_df)

# # 上传要预测的数据集
# st.write("Make predictions:")
# prediction_file = st.file_uploader("Upload a CSV file to make predictions", type="csv")
# if prediction_file is not None:
#     prediction_data = pd.read_csv(prediction_file)
#     predictions = model.predict(prediction_data[feature_variables])
#     prediction_data[target_variable] = predictions
#     st.write("Predictions:")
#     st.write(prediction_data)
#     prediction_data.to_csv("predictions.csv", index=False)
#     st.download_button(
#         label="Download predictions as CSV",
#         data=prediction_data.to_csv(index=False),
#         file_name="predictions.csv",
#         mime="text/csv"
#     )

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# 定义函数：读取数据集
def load_data():
    uploaded_file = st.file_uploader("上传数据集", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.warning("请上传数据集")
        return None


# 定义函数：模型参数设置
def set_params():
    params = {}
    params['objective'] = st.sidebar.selectbox('回归/分类', ['reg:squarederror', 'binary:logistic'])
    params['n_estimators'] = st.sidebar.slider('树的数量', min_value=10, max_value=200, value=100, step=10)
    params['max_depth'] = st.sidebar.slider('树的深度', min_value=1, max_value=20, value=3, step=1)
    params['learning_rate'] = st.sidebar.slider('学习率', min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    params['gamma'] = st.sidebar.slider('Gamma', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    params['min_child_weight'] = st.sidebar.slider('Min_child_weight', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    return params


# 定义函数：数据集划分
def split_data(df):
    y_col = st.sidebar.selectbox('选择目标变量', df.columns)
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    test_size = st.sidebar.slider('测试集比例', min_value=0.1, max_value=0.5, value=0.3, step=0.1)
    random_state = st.sidebar.slider('随机数种子', min_value=0, max_value=100, value=42, step=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# 定义函数：训练模型
def train_model(X_train, y_train, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    num_round = params['n_estimators']
    bst = xgb.train(params, dtrain, num_round)
    return bst


# 定义函数：模型评估
def eval_model(bst, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred_1 = bst.predict(dtest)
    y_pred = pd.DataFrame({'预测值': y_pred_1})
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write('Mean Squared Error:', round(mse, 2))
    return y_pred

# 定义函数：下载结果数据集
from io import BytesIO
def download_results(df_result):
    csv = df_result.to_csv(index=False).encode()
    b = BytesIO(csv)
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">下载结果数据集</a>'
    return href

# 上传数据集
df = load_data()

if df is not None:

    # 模型参数设置
    params = set_params()

    # 数据集划分
    X_train, X_test, y_train, y_test = split_data(df)

    # 训练模型
    bst = train_model(X_train, y_train, params)

    # 模型评估
    st.write('## 模型评估')
    df_result = eval_model(bst, X_test, y_test)
    st.write(df_result.head(10))
    st.write(f'Mean Squared Error: {y_pred["Mean Squared Error"][0]}')

    # 下载结果数据集
    st.markdown(download_results(df_result), unsafe_allow_html=True)
