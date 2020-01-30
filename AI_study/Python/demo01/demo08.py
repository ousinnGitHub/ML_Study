import numpy as np
import matplotlib.pyplot as plt

# 读取数据
train = np.loadtxt('click.csv',delimiter=',',dtype='int',skiprows=1)
train_x: list = train[:,0]
train_y: list = train[:,1]

# 标准化
mu = train_x.mean()
sigma = train_x.std()
def standize(x):
    return (x - mu) / sigma

train_z = standize(train_x)

# 参数初期化
theta0 = np.random.rand()
theta1 = np.random.rand()

# 预测函数
def f(x):
    return theta0 + theta1 * x

# 目的函数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 学习率
ETA = 1e-3

# 误差 差分
diff = 1

# 更新回数
count = 0

# 誤差の差分が0.01以下になるまでパラメータ更新を繰り返す
error = E(train_z, train_y)

while diff > 1e-2:
    # 更新结果一时保存
    tmp_theta0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp_theta1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)

    # 参数更新
    theta0 = tmp_theta0
    theta1 = tmp_theta1

    # 前回的误差计算
    current_error = E(train_z,train_y)
    diff = error - current_error
    error = current_error

    # log 出力
    count += 1
    log = '{}回目:theta0 = {:.3f},theta1 = {:.3f},差分 = {:.4f}'
    print(log.format(count,theta0,theta1,diff))

# 画图
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()

