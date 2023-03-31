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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# 添加上传数据集功能
uploaded_file = st.file_uploader("上传数据集", type=["csv"])

# 如果没有上传数据集，则提示用户上传
if uploaded_file is None:
    st.warning('请上传数据集')
else:
    # 加载数据集
    data = pd.read_csv(uploaded_file)

    # 显示数据集
    st.write('数据集：')
    st.write(data.head())

# 显示数据集的前5行
st.write("Data Preview:")
st.write(data.head())

# 选择目标变量和特征变量
target_variable = st.selectbox("Select target variable", data.columns)
feature_variables = st.multiselect("Select feature variables", data.columns)

# 拆分数据集为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 定义XGBoost模型参数
params = {
    "max_depth": st.slider("Maximum depth", 1, 10, 3),
    "learning_rate": st.slider("Learning rate", 0.01, 0.5, 0.1),
    "n_estimators": st.slider("Number of estimators", 10, 500, 100),
    "objective": "reg:squarederror"
}

# 训练XGBoost模型
X_train = train_data[feature_variables]
y_train = train_data[target_variable]
X_test = test_data[feature_variables]
y_test = test_data[target_variable]
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# 在测试集上评估模型性能
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write("Mean squared error:", mse)

# 显示特征重要性
st.write("Feature importances:")
importance_df = pd.DataFrame({
    "Feature": feature_variables,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
st.bar_chart(importance_df)

# 上传要预测的数据集
st.write("Make predictions:")
prediction_file = st.file_uploader("Upload a CSV file to make predictions", type="csv")
if prediction_file is not None:
    prediction_data = pd.read_csv(prediction_file)
    predictions = model.predict(prediction_data[feature_variables])
    prediction_data[target_variable] = predictions
    st.write("Predictions:")
    st.write(prediction_data)
    prediction_data.to_csv("predictions.csv", index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=prediction_data.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
