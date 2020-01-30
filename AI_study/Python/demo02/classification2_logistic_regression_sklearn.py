# sklearn 版

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('images2.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# 標準化
std = StandardScaler()
train_z = std.fit_transform(train_x)

# モデル作成(L2正規化、Cが大きくなると重みも大きくなる)
model = linear_model.LogisticRegression(C=10)

# モデル訓練
model.fit(train_z, train_y)

# パラメータを初期化
theta = np.random.rand(3)

print(model.coef_)

theta[0] = model.intercept_[0]
theta[1] = model.coef_[0, 0]
theta[2] = model.coef_[0, 1]

print(theta[0])  # 傾き
print(theta[1])  # 切片
print(theta[2])  # 切片

print(model.get_params())  # パラメータの取得

# 境界線の式
#   theta1・x + theta2・y + theta0 = 0
#   ⇒ y = (-theta1・x - theta0) / theta2

# プロットして確認
x0 = np.linspace(-2, 2, 100)
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle='dashed')
plt.show()
