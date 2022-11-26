import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 项目根路径
project_root_path = os.path.dirname(os.path.dirname(__file__))
# 导入数据
df = pd.read_csv(os.path.join(project_root_path, 'data/shujuji.csv'))
columns_name = df.columns.values  # 获取所有列名
Y_name = 'classification'  # 获取目标Y列名，需要自己定义哪一行为Y
X_name = [name for name in columns_name if name != Y_name]  # 获取特征X列名
X, Y = df[X_name].values, df[Y_name].values  # 获取特征X和目标Y数据
# 特征归一化
scaler = MinMaxScaler()  # 实例化
norm_X = scaler.fit_transform(X)  # 得到归一化后的X

df[X_name] = norm_X
df_coor = df.corr(method='pearson')
fig, ax = plt.subplots(figsize=(16, 12), facecolor='w')
# 指定颜色带的色系
sns.heatmap(df_coor, annot=True, vmin=-1, vmax=1, square=True, cmap="rainbow", fmt='.2f', annot_kws={'size': 4})  # annot=True显示相关性大小
# plt.title('相关性热力图')
plt.show()
fig.savefig('output/corr.png', transparent=False)
