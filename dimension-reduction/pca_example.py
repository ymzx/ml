# -*- coding:utf-8 -*-
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

x2, y2 = make_moons(n_samples=100, random_state=123)

if __name__ == '__main__':
    pca = PCA(n_components=2)
    x2_std = pca.fit_transform(x2)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    ax[0].scatter(x2_std[y2 == 0, 0], x2_std[y2 == 0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(x2_std[y2 == 1, 0], x2_std[y2 == 1, 1], color='blue', marker='o', alpha=0.5)
    plt.scatter(x2_std[y2 == 0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
    plt.scatter(x2_std[y2 == 1, 0], np.zeros((50, 1)) + 0.02, color='blue', marker='o', alpha=0.5)
    ax[1].set_yticks([])
    ax[0].set_ylabel('PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()