import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

train = np.loadtxt('click.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

std = StandardScaler()
train_z = std.fit_transform(train_x.reshape(-1, 1))
print(train_z)

model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(train_z)
model.fit(X_poly, train_y)

print(X_poly)
print(model.coef_)
print(model.intercept_)
print(model.get_params())

x = np.linspace(-4, 4, 100)
y = model.predict(poly_reg.fit_transform(x.reshape(-1, 1)))

plt.plot(train_z, train_y, 'o')
plt.plot(x, y)
plt.show()
