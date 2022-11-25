from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

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

# randromforest train and predict
rf = RandomForestClassifier(random_state=0, max_depth=10)


# train
rf.fit(x_train, y_train)

# 使用训练的模型来预测train数据
y_train_pred = rf.predict(x_train)
y_train_pred = np.array(y_train)
recall = recall_score(y_train, y_train_pred, average="macro")

# 使用训练的模型来预测test数据
y_test_pred = rf.predict(x_test)
y_test_pred = np.array(y_test_pred)
recall = recall_score(y_test, y_test_pred, average="macro")
print('test recall', recall)
