import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
import sklearn.svm as svm
import os
# 项目根路径
project_root_path = os.path.dirname(os.path.dirname(__file__))
# 导入数据
df = pd.read_csv(os.path.join(project_root_path, 'data/shujuji.csv'))

columns_name = df.columns.values  # 获取所有列名
Y_name = 'classification'  # 获取目标Y列名，需要自己定义哪一行为Y
X_name = [name for name in columns_name if name != Y_name]  # 获取特征X列名
X, Y = df[X_name].values, df[Y_name].values  # 获取特征X和目标Y数据
Y -= 1  # label索引必须从0开始，label must be in [0, num_class).
class_num = len(set(Y.tolist()))  # 类别数

# 划分训练和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# 归一化
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# svm
model = svm.SVC(kernel="linear", decision_function_shape="ovo", random_state=42)  # 核函数， "linear", "poly", "rbf", "sigmoid"
model.fit(x_train, y_train)

# 评测
acu_train = model.score(x_train, y_train)
acu_test = model.score(x_test, y_test)

# 预测
y_pred = model.predict(x_test)
recall = recall_score(y_test, y_pred, average="macro")

print(acu_train, acu_test, recall)

# 模型保存与加载

