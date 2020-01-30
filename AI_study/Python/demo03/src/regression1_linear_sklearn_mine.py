import numpy as np
import matplotlib.pyplot as plt
from numpy.core._multiarray_umath import ndarray
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

train: ndarray = np.loadtxt('click.csv', dtype='int', delimiter=',', skiprows=1)
train_x: ndarray = train[:, 0]
train_y: ndarray = train[:, 1]

# 标准化
std = StandardScaler()
train_z = std.fit_transform(train_x.reshape(-1, 1))

# model 做成
model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
# 训练
model.fit(train_z, train_y)

# 斜率
print(model.coef_)
# 切片
print(model.intercept_)
# 参数
print(model.get_params())

x = np.linspace(-3, 3, 100)
y = model.predict(x.reshape(-1, 1))

x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, y)
plt.show()