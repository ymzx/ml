# coding=utf-8
from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
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
kpca = KernelPCA(kernel='rbf', gamma=10, n_components=2)
newMat = kpca.fit_transform(norm_X)

