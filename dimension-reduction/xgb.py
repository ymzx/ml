import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os

# 项目根路径
project_root_path = os.path.dirname(os.path.dirname(__file__))
# 导入数据
df = pd.read_csv(os.path.join(project_root_path, 'data/shujuji.csv'))

columns_name = df.columns.values  # 获取所有列名
Y_name = 'classification'  # 获取目标Y列名，需要自己定义哪一行为Y
X_name = [name for name in columns_name if name != Y_name]  # 获取特征X列名
X, Y = df[X_name].values, df[Y_name].values  # 获取特征X和目标Y数据

# fit model no training data
xgb = XGBClassifier()
xgb.fit(X, Y)
# feature importance
print(xgb.feature_importances_)
# plot
plt.barh(range(len(xgb.feature_importances_)), xgb.feature_importances_)
plt.yticks(range(len(xgb.feature_importances_)), X_name)
plt.show()