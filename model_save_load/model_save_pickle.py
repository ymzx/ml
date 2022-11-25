from sklearn import svm
from sklearn import datasets
import pickle
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y) # 训练
# 保存模型
model_save_path = 'model_files/svm2.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump(clf, f)

# --------------------模型加载和推理-----------------------------------#
# 模型加载
with open(model_save_path, 'rb') as f:
    model = pickle.load(f)
# 推理测试
input = X[0:1]
out = model.predict(input)
print('预测结果：', out)


