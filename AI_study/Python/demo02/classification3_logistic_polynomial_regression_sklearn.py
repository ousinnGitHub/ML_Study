# ######################
#  パラメータ検証

import numpy as np
import sympy as sympy
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('data3.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:,0:2]
train_y = train[:,2]

# 標準化
std = StandardScaler()
train_z = std.fit_transform(train_x)

# モデル作成(L2正規化、Cが正規化パラメータです。Cが大きくなると重みも大きくなる)
model = linear_model.LogisticRegression(C=1.0)

# 多項式変換
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(train_z)
#[1, a, b, a ^ 2, ab, b ^ 2].

print(X_poly)
# モデル訓練
model.fit(X_poly, train_y)

print(model.coef_)
# パラメータを初期化
theta = np.random.rand(6)

theta[0] = model.coef_[0,0]
theta[1] = model.coef_[0,1]
theta[2] = model.coef_[0,2]
theta[3] = model.coef_[0,3]
theta[4] = model.coef_[0,4]
theta[5] = model.coef_[0,5]

print(model.get_params())  #パラメータの取得

# 境界線の式
#   theta1・x + theta2・y + theta0 = 0
#   ⇒ y = (-theta1・x - theta0) / theta2

#def solve(a, b, c):
#    return print(sympy.solve(2,-3,2))

#[1, a, b, a ^ 2, ab, b ^ 2].

# プロットして確認
x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, linestyle='dashed')
plt.show()
