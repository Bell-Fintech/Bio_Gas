import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

diagrams_df = pd.read_csv('../data/economy_data.csv', index_col=None)

df = diagrams_df.copy()

# Separating X and y
X = df.drop(['amount_of_electricity_fed_into_the_grid'], axis=1)
Y = df['amount_of_electricity_fed_into_the_grid']

std_x = StandardScaler()

X = std_x.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 4)


regressor = LinearRegression()
"""
    LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):线性回归
        parameters：
            fit_intercept: 是否计算截距
            normalize:     是否规范化
            copy_x:        是否复制x
        Attributes:
            coef_:         系数
            intercept_:    截距
        Methods:
            fit(X, y, sample_weight=None):   拟合
            get_params(deep=True):           得到参数,如果deep为True则得到这个estimator 的子对象返回参数名和值的映射
            set_params(**params):            设置参数
            predict(X):                      预测
            score(X, y, sample_weight=None): 预测的准确度。X：测试样本；y：X的真实结果；sample_w
            eight:                           权重
"""
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

print("MAE:", mean_absolute_error(Y_test,Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R2:", r2_score(Y_test, Y_pred))

# Saving the model
import pickle
pickle.dump(regressor, open('../models/economy_amount_of_electricity_fed_into_the_grid_predict.pkl', 'wb'))