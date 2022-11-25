import pandas as pd
import xgboost as xgb
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
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

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=100)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test, label=y_test)

# 设置模型参数
params = {
    'objective': 'multi:softmax',
    'eta': 0.1,
    'max_depth': 2,
    'num_class': class_num
}

watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
# 设置训练轮次
num_round = 100
bst = xgb.train(params, xgb_train, num_round, watchlist)

# 模型预测
pred = bst.predict(xgb_test)
print(pred)

#模型评估
error_rate = np.sum(pred != y_test) / y_test.shape[0]
print('测试集错误率(softmax):{}'.format(error_rate))

accuray = 1 - error_rate
print('测试集准确率：%.4f' % accuray)

# 模型保存
bst.save_model("model_files/000.model")

# 模型加载
bst = xgb.Booster()
bst.load_model("model_files/000.model")
pred=bst.predict(xgb_test)
print(pred)
