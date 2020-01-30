import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

train = np.loadtxt('click.csv', delimiter=',', dtype='int', skiprows=1)
X = train[:, 0]
Y = train[:, 1]

X_train, X_Test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=2),
                      linear_model.SGDRegressor(max_iter=1000)
                      )

model.fit(X_train.reshape(-1, 1), Y_train)

print("PARAM:", model.get_params())

Y_pred = model.predict(X_Test.reshape(-1, 1))

from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(Y_test, Y_pred)

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(Y_test, Y_pred)

RMSE = np.sqrt(MSE)

x = np.linspace(0, 300, 100)
y = model.predict(x.reshape(-1, 1))

plt.plot(X_train, Y_train, 'o', label='train data')
plt.plot(x, y, 'r-')
plt.plot(X_Test, Y_test, 'x', label='test data')
plt.text(10, 600, 'MAE:%.2f' % MAE)
plt.text(10, 570, 'MSE:%.2f' % MSE)
plt.text(10, 540, 'RISE:%.2f' % RMSE)
plt.legend()
plt.show()
