"""
in terminal run: streamlit run main.py
in another terminal run: mlflow ui
"""
# import time
#
# import streamlit as st
# import pandas as pd
# from PIL import Image
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
# import matplotlib.pyplot as plt
# import os
# import pycaret.classification as pc_cl
# import pycaret.regression as pc_rg
# import mlflow
# import BiLSTM_VAE_WOA.bilstm_vae_woa as bilstm_vae_woa
# from c_e_models.model import main as ce_main
# from c_e_models import loss
#
#
# def get_model_training_logs(n_lines=10):
#     file = open('logs.log', 'r')
#     lines = file.read().splitlines()
#     file.close()
#     return lines[-n_lines:]
#
#
# CHART_LIST = ['折线图', '直方图', '饼图']
# ML_TASK_LIST = ['回归', '分类', '自定义模型']
# RG_MODEL_LIST = ['lr', 'svm', 'rf', 'xgboost', 'lightgbm']
# CL_MODEL_LIST = ['lr', 'dt', 'svm', 'rf', 'xgboost', 'lightgbm']
# DE_MODEL_LIST = ['BiLstm_VAE_WOA', 'ITCADenseNet_DITCANet']
#
#
# def list_files(directory, extension):
#     # list certain extension files in the folder
#     return [f for f in os.listdir(directory) if f.endswith('.' + extension)]
#
#
# def concat_file_path(file_folder, file_selected):
#     # handle the folder path with '/' or 'without './'
#     # and concat folder path and file path
#     if str(file_folder)[-1] != '/':
#         file_selected_path = file_folder + '/' + file_selected
#     else:
#         file_selected_path = file_folder + file_selected
#     return file_selected_path
#
#
# @st.cache(suppress_st_warning=True)
# def load_csv(file_selected_path, nrows):
#     # load certain rows
#     try:
#         if nrows == -1:
#             df = pd.read_csv(file_selected_path)
#         else:
#             df = pd.read_csv(file_selected_path, nrows=nrows)
#     except Exception as ex:
#         df = pd.DataFrame([])
#         st.exception(ex)
#     return df
#
#
# def app_main():
#     st.set_page_config(  # 设置页面格式
#         page_title="Ex-stream-ly Cool App",
#         # page_icon="🧊",
#         layout="wide",  # "centered",
#         initial_sidebar_state="expanded",
#         menu_items={
#             'Get Help': 'https://www.extremelycoolapp.com/help',
#             'Report a bug': "https://www.extremelycoolapp.com/bug",
#             'About': "# This is a header. This is an *extremely* cool app!"
#         }
#     )
#     st.title("自动化机器学习平台")  # 设置页面标题
#     # st.snow()
#     # 设置页面主图片
#     image = Image.open('./picture/1648433411.png')
#     st.image(image, width=750)  # , caption='自动化机器学习平台'
#     placeholder = st.empty()
#
#     # 设置按钮回掉函数功能
#     def start1():
#         st.write(f'npops', st.session_state.npops)
#         st.write(f'ngens', st.session_state.ngens)
#         st.success(f'数据选取完成')
#         st.success(f'训练模型中。。。')
#         bilstm_vae_woa.start_detector()
#         st.success(f'模型预测完毕。。。')
#
#     # 设置按钮回掉函数功能
#     def start2():
#         st.write(f'npops', st.session_state.npops)
#         st.write(f'ngens', st.session_state.ngens)
#         st.success(f'数据选取完成')
#         st.success(f'训练模型中。。。')
#         time.sleep(30)
#         loss.main()
#         st.success(f'模型训练完毕。。。')
#         st.success(f'读取预测数据。。。')
#         ce_main()
#         st.success(f'模型预测完毕。。。')
#
#     # 设置数据选取模块
#     if st.sidebar.checkbox('定义数据源'):
#         placeholder.empty()
#         file_folder = st.sidebar.text_input('文件夹', value="data")
#         data_file_list = list_files(file_folder, 'csv')
#         if len(data_file_list) == 0:
#             st.warning(f'当路径无可用数据集')
#         else:
#             file_selected = st.sidebar.selectbox(
#                 '选择文件', data_file_list)
#             file_selected_path = concat_file_path(file_folder, file_selected)
#             nrows = st.sidebar.number_input('行数', value=-1)
#             n_rows_str = '全部' if nrows == -1 else str(nrows)
#             with placeholder.container():
#                 st.info(f'已选择文件：{file_selected_path}，读取行数为{n_rows_str}')
#                 df = load_csv(file_selected_path, nrows)
#                 st.table(df)
#     else:
#         file_selected_path = None
#         nrows = 100
#         st.warning(f'当前选择文件为空，请选择。')
#
#     # 设置数据分析模块
#     if st.sidebar.checkbox('数据分析'):
#         placeholder.empty()
#         if file_selected_path is not None:
#             df = load_csv(file_selected_path, nrows)
#             if st.sidebar.button('一键生成数据探索性分析报告'):
#                 pr = ProfileReport(df, explorative=True)
#                 st_profile_report(pr)
#             try:
#                 cols = df.columns.to_list()
#                 target_col = st.sidebar.selectbox('选取展示数据对象', cols)
#             except BaseException:
#                 st.sidebar.warning(f'数据格式无法正确读取')
#                 target_col = None
#             visualization = st.sidebar.selectbox('Select a Chart type', CHART_LIST)
#             with placeholder.container():
#                 if visualization == "折线图":
#                     fig = plt.figure(figsize=(20, 7))
#                     plt.plot(df[target_col], color='r', ls='--', label='预测值')
#                     plt.legend()
#                     plt.show()
#                     st.pyplot(fig)
#                 elif visualization == "直方图":
#                     fig = plt.figure(figsize=(20, 7))
#                     plt.hist(df[target_col], bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
#                     # 显示横轴标签
#                     plt.xlabel("区间")
#                     # 显示纵轴标签
#                     plt.ylabel("频数/频率")
#                     # 显示图标题
#                     plt.title("频数/频率分布直方图")
#                     plt.show()
#                     st.pyplot(fig)
#
#         else:
#             st.info(f'没有选择文件，无法进行分析。')
#
#     # 设置模型训练模块
#     if st.sidebar.checkbox('快速建模'):
#         placeholder.empty()
#         if file_selected_path is not None:
#             task = st.sidebar.selectbox('选择任务', ML_TASK_LIST)
#             if task == '回归':
#                 model = st.sidebar.selectbox('选取模型', RG_MODEL_LIST)
#             elif task == '分类':
#                 model = st.sidebar.selectbox('选取模型', CL_MODEL_LIST)
#             elif task == '自定义模型':
#                 model = st.sidebar.selectbox('选取模型', DE_MODEL_LIST)
#             df = load_csv(file_selected_path, nrows)
#             try:
#                 cols = df.columns.to_list()
#                 target_col = st.sidebar.selectbox('选取预测/检测对象', cols)
#             except BaseException:
#                 st.sidebar.warning(f'数据格式无法正确读取')
#                 target_col = None
#
#             if target_col is not None and st.sidebar.button('训练模型'):
#                 if task == '回归':
#                     st.success(f'数据预处理。。。')
#                     pc_rg.setup(
#                         df,
#                         target=target_col,
#                         log_experiment=True,
#                         experiment_name='ml_',
#                         log_plots=True,
#                         silent=True,
#                         verbose=False,
#                         profile=True)
#                     st.success(f'数据预处理完毕。')
#                     st.success(f'训练模型。。。')
#                     pc_rg.create_model(model, verbose=False)
#                     st.success(f'模型训练完毕。。。')
#                     # pc_rg.finalize_model(model)
#                     st.success(f'模型已经创建')
#                 elif task == '分类':
#                     st.success(f'数据预处理。。。')
#                     pc_cl.setup(
#                         df,
#                         target=target_col,
#                         fix_imbalance=True,
#                         log_experiment=True,
#                         experiment_name='ml_',
#                         log_plots=True,
#                         silent=True,
#                         verbose=False,
#                         profile=True)
#                     st.success(f'数据预处理完毕。')
#                     st.success(f'训练模型。。。')
#                     pc_cl.create_model(model, verbose=False)
#                     st.success(f'模型训练完毕。。。')
#                     # pc_cl.finalize_model(model)
#                     st.success(f'模型已经创建')
#                 elif task == '自定义模型':
#                     if model == 'BiLstm_VAE_WOA':
#                         with placeholder.container():
#                             npops = st.number_input(label='Enter 种群数量', value=50, key='npops',
#                                                     help='number of solutions per generation')
#                             ngens = st.number_input(label='Enter 迭代次数', value=100, key='ngens',
#                                                     help='number of generations')
#                             ndim = st.number_input(label='Enter 基检测器数量', value=30, key='ndim',
#                                                    help='鲸鱼个体的编码/维度长度')
#                             a = st.number_input(label='Enter 控制搜索的速度与范围值', value=2.00, key='a',
#                                                 help='woa algorithm specific parameter')
#                             b = st.number_input(label='Enter 螺旋参数controls spiral', value=0.50, key='b',
#                                                 help='woa algorithm specific parameter')
#                             c0 = st.number_input(label='Enter 绝对解约束值', value=-0.010, key='c0',
#                                                  help='权重的最小取值')
#                             c1 = st.number_input(label='Enter 绝对解约束值', value=0.150, key='c1',
#                                                  help='权重的最大取值')
#                             st.button('提交数据', on_click=start1)
#                     elif model == 'ITCADenseNet_DITCANet':
#                         with placeholder.container():
#                             npops = st.number_input(label='Enter 迭代次数', value=200, key='npops')
#                             ngens = st.number_input(label='Enter batch_size', value=16, key='ngens')
#                             ndim = st.number_input(label='Enter 模型深度', value=15, key='ndim')
#                             a = st.number_input(label='Enter 模型宽度', value=5, key='a')
#                             c0 = st.number_input(label='Enter 学习率', value=-0.001, key='c0')
#                             c1 = st.number_input(label='Enter 蒸馏系数', value=0.20, key='c1')
#                             st.button('提交数据', on_click=start2)
#     # 设置模型应用模块
#     if st.sidebar.checkbox('查看系统日志'):
#         n_lines = st.sidebar.slider(label='行数', min_value=3, max_value=50)
#         if st.sidebar.button("查看"):
#             logs = get_model_training_logs(n_lines=n_lines)
#             st.text('系统日志')
#             st.write(logs)
#     try:
#         all_runs = mlflow.search_runs(experiment_ids=0)
#     except:
#         all_runs = []
#     if len(all_runs) != 0:
#         if st.sidebar.checkbox('预览模型'):
#             ml_logs = 'http://kubernetes.docker.internal:5000/  -->开启mlflow，命令行输入:mlflow ui'
#             st.markdown(ml_logs)
#             st.dataframe(all_runs)
#         if st.sidebar.checkbox('选择模型'):
#             selected_run_id = st.sidebar.selectbox('从已保存模型中选择',
#                                                    all_runs[all_runs['tags.Source'] == 'create_model'][
#                                                        'run_id'].tolist())
#             selected_run_info = all_runs[(
#                     all_runs['run_id'] == selected_run_id)].iloc[0, :]
#             st.code(selected_run_info)
#             if st.sidebar.button('预测数据'):
#                 model_uri = f'runs:/' + selected_run_id + '/model/'
#                 model_loaded = mlflow.sklearn.load_model(model_uri)
#                 df = pd.read_csv(file_selected_path, nrows=nrows)
#                 # st.success(f'模型预测中。。。   ')
#                 pred = model_loaded.predict(df)
#                 pred_df = pd.DataFrame(pred, columns=['预测值'])
#                 st.dataframe(pred_df)
#                 pred_df.plot()
#                 st.pyplot()
#     else:
#         st.sidebar.warning('没有找到训练好的模型')
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

st.title("To Be Continued......")
import streamlit as st
import pandas as pd

d1 = ['CSTR', 'USR', 'UASB' ]
df1 = pd.DataFrame(data=d1)

selection =  st.selectbox('发酵工艺:', ['CSTR', 'USR', 'UASB'])

for i in d1:
    if selection==d1[0]:
        id_selected=0
    elif selection==d1[1]:
        id_selected = 1
    else:
        id_selected = 2
result=id_selected


import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
def main():
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

if __name__ == "__main__":
    main()
