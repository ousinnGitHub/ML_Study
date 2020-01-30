# sklearn 版

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('click.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

# 標準化
std = StandardScaler()
train_z = std.fit_transform(train_x.reshape(-1, 1))

# モデル作成
model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                                      normalize=False)

# 多项式モデル訓練
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(train_z)
model.fit(X_poly, train_y)

print(X_poly)
print(model.coef_)  # 傾き
print(model.intercept_)  # 切片
print(model.get_params())  # パラメータの取得

# モデル推論
x = np.linspace(-3, 3, 100)
y = model.predict(poly_reg.fit_transform(x.reshape(-1, 1)))

# プロットして確認
plt.plot(train_z, train_y, 'o')
plt.plot(x, y)
plt.show()
