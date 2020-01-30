# sklearn 版

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('click.csv', delimiter=',', dtype='int', skiprows=1)
X = train[:,0]
Y = train[:,1]

# テストデータと訓練データの抽出
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

# piplineでモデル作成
model = make_pipeline(StandardScaler(), #標準化
              PolynomialFeatures(degree=2), # 2次多项式
              linear_model.SGDRegressor(max_iter=1000) #iterationの回数を1000にする
              )
# 多项式モデル
# model:
# y = θ0 + θ1 * x  + θ02 * x **2

# 訓練
model.fit(X_train.reshape(-1,1), Y_train)

# モデルパラメータ
print("PARAM:", model.get_params())  #パラメータの取得

# モデル評価
Y_pred =  model.predict(X_test.reshape(-1,1))

#平均絶対誤差 (MAE)
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(Y_test, Y_pred)

#平均二乗誤差 (MSE)
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(Y_test, Y_pred)

#二乗平均平方根誤差 (RMSE)
RMSE = np.sqrt(MSE)

# モデル推論(モデルをプロット用データ生成)
x = np.linspace(0, 300, 100)
y = model.predict(x.reshape(-1,1))

# プロットして確認
plt.plot(X_train, Y_train, 'o',label='train data')
plt.plot(x, y,'r-')
plt.plot(X_test, Y_test, 'x',label='test data')
plt.text(10,600,'MAE:%.2f'%MAE)
plt.text(10,570,'MSE:%.2f'%MSE)
plt.text(10,540,'RMSE:%.2f'%RMSE)
plt.legend()
plt.show()