from sklearn import svm
from sklearn import datasets
import joblib
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
model_save_path = 'model_files/svm.pkl'
# 保存模型
joblib.dump(clf, model_save_path)

# -------------------------------------------------------#
# 加载模型并预测
clf = joblib.load(model_save_path)
# 推理预测
input = X[0:1]
out = clf.predict(input)
print('预测结果：', out)
