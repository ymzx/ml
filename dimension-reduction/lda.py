import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys, os
import numpy as np
# 项目根路径
project_root_path = os.path.dirname(os.path.dirname(__file__))
# 导入数据
df = pd.read_csv(os.path.join(project_root_path, 'data/shujuji.csv'))
columns_name = df.columns.values  # 获取所有列名
Y_name = 'classification'  # 获取目标Y列名，需要自己定义哪一行为Y
X_name = [name for name in columns_name if name != Y_name]  # 获取特征X列名
X, Y = df[X_name].values, df[Y_name].values  # 获取特征X和目标Y数据
# LDA降维
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, Y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=Y)
plt.show()

# 将新输入x降维
nrow, ncol = X.shape  # 行与列
m, n = 1, ncol
x = np.random.rand(m, n)  # 随机生成0-1间m*n的矩阵，并作为新的输入x
newx = lda.transform(x)
print(x, newx)
