import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn import linear_model,model_selection,metrics
from sklearn.pipeline import make_pipeline

# 学習データを読み込む(第一行はskipされる)
train = np.loadtxt('wine.csv', delimiter=',', skiprows=0)
X = train[:,0:2]
Y = train[:,2]

# テストデータと訓練データの抽出
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=0, shuffle=True)

# piplineでモデル作成
model = make_pipeline(StandardScaler(), #標準化
              #PolynomialFeatures(degree=2), # 2次多项式
              linear_model.SGDClassifier(loss="log")) # loss="hinge", loss="log"

# モデル訓練
model.fit(X_train, Y_train)

#x1 = np.linspace(0, 300, 100)
x1= X
y1 = model.decision_function(X);

# モデル
#x1 = np.linspace(0, 300, 100)
#x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]

plt.plot(X_train[Y_train == 1, 0],X_train[Y_train == 1, 1], 'ro',label='train:cat1')
plt.plot(X_train[Y_train == 2, 0],X_train[Y_train == 2, 1], 'rx',label='train:cat2')
plt.plot(X_test[Y_test == 1, 0],X_test[Y_test == 1, 1], 'bo',label='test:cat1')
plt.plot(X_test[Y_test == 2, 0],X_test[Y_test == 2, 1], 'bx',label='test:cat2')
plt.legend()
plt.show()


# テストデータ正答率を求める
Y_pred = model.predict(X_test)

###########################
# モデル評価指標
###########################

# 全体正解率
ac_score = metrics.accuracy_score(Y_test, Y_pred)
print("正确率 = ", ac_score)

# クラス1が正例の場合
precision_score = metrics.precision_score(Y_test, Y_pred)
print("精度 = ", precision_score)

recall_score = metrics.recall_score(Y_test, Y_pred)
print("召回率 = ", recall_score)

f1 = metrics.f1_score(Y_test, Y_pred)
print("F1值=", f1)

# クラス1とクラス2がそれぞれ正例の場合
report = metrics.classification_report(Y_test,Y_pred)
print("Report=", report)

###########################
# PR曲線(Pricesion, Recall)
###########################

Y_probas_pred = model.predict_proba(X_test) # 予測率

# precision,recall, しきい値 を計算
precision, recall, thresholds = metrics.precision_recall_curve(Y_test, Y_probas_pred[:,0:1], pos_label=1)#　クラス1は正例

auc = metrics.auc(recall, precision)

# PR曲線のプロット
plt.plot(recall, precision,label='PR curve (area = %.2f)'%auc)
plt.legend()
plt.title('PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

##########################
# ROC曲線
##########################

# FPR, TPR(, しきい値) を算出
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_probas_pred[:,0:1], pos_label=1)

# AUC算出
auc = metrics.auc(fpr, tpr)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()
