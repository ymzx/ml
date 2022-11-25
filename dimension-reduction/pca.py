import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import sys, os
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
# inv_norm_X = scaler.inverse_transform(norm_X) # 将归一化结果逆转得到原始数据
# 特征降维-PCA(主成分分析)
pca = PCA(n_components=2)  # 保留主成分个数或特征数 n_components
newX = pca.fit_transform(norm_X)  # 得到降维后的数据
invX = pca.inverse_transform(newX)  # 将降维后的数据转换成原始数据
pca_ratio = pca.explained_variance_ratio_  # 各主成分方差所占百分比
eigenvector = pca.components_  # 输出特征向量
# 将新输入x降维
nrow, ncol = norm_X.shape  # 行与列
m, n = 1, ncol
x = np.random.rand(m, n)  # 随机生成0-1间m*n的矩阵，并作为新的输入x
newx = pca.transform(x)
print(x, newx, pca_ratio)

# https://blog.csdn.net/weixin_40637477/article/details/124609872
