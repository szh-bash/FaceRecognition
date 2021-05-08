import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, IncrementalPCA

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from test.r50Feat import get
from init import DataReader
from config import modelPath, dataPath
from math import sqrt


def sqr(a):
    return a*a


if __name__ == '__main__':
    test_data = 'faces96C'
    filepath = dataPath[test_data]
    print(filepath)
    data = DataReader('test', test_data)
    store, feats, sps = get(modelPath, data)
    print(len(store))
    print(len(sps))
    val = np.random.rand(len(sps))
    X = []
    col = []
    name = []
    cnt = 0
    up = 6
    cls_col = list(range(0, up))
    for cls in sps:
        if cnt == up:
            break
        for x in sps[cls]:
            ft = store[x]
            ft -= np.mean(ft)
            ft /= np.std(ft)
            X.append(ft)
            col.append(cls_col[cnt])
        cnt += 1
        name.append(cls)
    print(name)
    X = np.array(X, dtype=np.float64)
    print(X.shape)
    pca = IncrementalPCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    X_new = pca.transform(X)
    X_new[:, 0] -= np.mean(X_new[:, 0])
    X_new[:, 0] /= np.std(X_new[:, 0])
    X_new[:, 1] -= np.mean(X_new[:, 1])
    X_new[:, 1] /= np.std(X_new[:, 1])

    # lth = []
    # for i in range(X_new.shape[0]):
    #     lth.append(sqrt(sqr(X_new[i, 0])+sqr(X_new[i, 1]))/10)
    # plt.scatter(X_new[:, 0]/lth[:], X_new[:, 1]/lth[:], marker='o', c=col)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=col)
    plt.title(test_data)
    plt.show()
