#CrossVal(cross_val_score).py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn import linear_model,model_selection
from sklearn.pipeline import make_pipeline

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('wine.csv', delimiter=',', skiprows=0)
X = train[:,0:2]
Y = train[:,2]

#多次元数
k_range = range(0, 10)

#得点
k_scores = []

# 由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    # piplineでモデル作成
    model = make_pipeline(StandardScaler(),  # 標準化
                          PolynomialFeatures(degree=k), # 2次多项式 超级参数
                          linear_model.SGDClassifier(loss="log"))  # loss="hinge", loss="log"

    scores = model_selection.cross_val_score(model, X, Y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

#可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for PolynomialFeatures')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
