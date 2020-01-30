# sklearn 版

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('click.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

# 標準化
std = StandardScaler()
train_z = std.fit_transform(train_x.reshape(-1,1))

# モデル作成
model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)

# モデル訓練
model.fit(train_z, train_y)

print(model.coef_) #傾き
print(model.intercept_) #切片
print(model.get_params())  #パラメータの取得

# モデル推論
x = np.linspace(-3, 3, 100)
y = model.predict(x.reshape(-1,1))

# プロットして確認
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, y)
plt.show()

