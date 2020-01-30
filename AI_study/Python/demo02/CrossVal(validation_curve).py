#CrossVal(validation_curve).py

# ######################################
# Overfitting 問題解決
# ######################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn import linear_model,model_selection

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('wine.csv', delimiter=',', skiprows=0)
X = train[:,0:2]
Y = train[:,2]

#建立参数测试集
param_range = range(1, 20)

#使用validation_curve快速找出参数对模型的影响
train_score, test_score = model_selection.validation_curve(
    linear_model.LogisticRegression(), X, Y, param_name='C', param_range=param_range, cv=10) #Cは正則化のパラメータ

#平均精度の平均値
train_loss_mean = np.mean(train_score, axis=1)
test_loss_mean = np.mean(test_score, axis=1)

#可视化图形
plt.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("C Param")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()
