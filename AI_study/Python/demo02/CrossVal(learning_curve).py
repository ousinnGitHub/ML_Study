
#CrossVal(learning_curve).py

# #############################
# Learning curve Overfittingチェック
# #############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn import linear_model,model_selection
from sklearn.pipeline import make_pipeline

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('wine.csv', delimiter=',', skiprows=0)
X = train[:,0:2]
Y = train[:,2]

#使用validation_c速找出参数对模型的影响urve快
train_sizes,train_loss, test_loss = model_selection.learning_curve(
    linear_model.LogisticRegression(C=8.0), X, Y, cv=10, train_sizes=[0.5, 0.6, 0.7, 0.8, 0.9, 1])

#平均每一轮的平均方差
train_loss_mean = np.mean(train_loss, axis=1)
test_loss_mean = np.mean(test_loss, axis=1)

#可视化图形
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

