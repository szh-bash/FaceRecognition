import pandas as pd
import numpy as np


def relu(x):
    #return np.maximum(x, 0)
    s = 1 / (1 + np.exp(-x))
    return s


# 有向图的邻接矩阵表征
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)
print("A = \n", A)
'''
[[0. 1. 0. 0.]
 [0. 0. 1. 1.]
 [0. 1. 0. 0.]
 [1. 0. 1. 0.]]
'''
# 两个整数特征
X = np.matrix([
    [i, -i]
    for i in range(A.shape[0])],
    dtype=float
)
print("X = \n", X)
'''
[[ 0.  0.]
 [ 1. -1.]
 [ 2. -2.]
 [ 3. -3.]]
'''
# 应用传播规则 A*X
# 每个节点的表征（每一行）现在是其相邻节点特征的和！
# 换句话说，图卷积层将每个节点表示为其相邻节点的聚合。
print("A * X = \n", A * X)
'''
[[ 1. -1.]
 [ 5. -5.]
 [ 1. -1.]
 [ 2. -2.]]
'''

# 上面存在的问题：
# 1、节点的聚合表征不包含它自己的特征！
# 2、度大的节点在其特征表征中将具有较大的值，度小的节点将具有较小的值。
#    这可能会导致梯度消失或梯度爆炸
I = np.matrix(np.eye(A.shape[0]))
print("I = \n", I)
'''
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
'''
# 每个节点都是自己的邻居
A_hat = A + I
print("A_hat = \n", A_hat)
'''
[[1. 1. 0. 0.]
 [0. 1. 1. 1.]
 [0. 1. 1. 0.]
 [1. 0. 1. 1.]]
'''
# 由于每个节点都是自己的邻居，每个节点在对相邻节点的特征求和过程中也会囊括自己的特征！
print("A_hat * X = \n", A_hat * X)
'''
[[ 1. -1.]
 [ 6. -6.]
 [ 3. -3.]
 [ 5. -5.]]
'''
# 为了防止某些度很大的节点的特征值很大，对特征表征进行归一化处理
# 通过将邻接矩阵 A 与度矩阵 D 的逆相乘，对其进行变换，从而通过节点的度对特征表征进行归一化
# f(X, A) = D⁻¹AX

# 首先计算出节点的度矩阵，这里改成了出度
print("A = \n", A)
D = np.array(np.sum(A, axis=1))[:, 0]
#D = np.array(np.sum(A, axis=0))[0]
D = np.matrix(np.diag(D))
print("D = \n", D)
'''
 [[1. 0. 0. 0.]
 [0. 2. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 2.]]
'''

# A归一化后,邻接矩阵中每一行的权重（值）都除以该行对应节点的度
print("D⁻¹ * A = \n", D**-1 * A)
'''
 [[0.  1.  0.  0. ]
 [0.  0.  0.5 0.5]
 [0.  1.  0.  0. ]
 [0.5 0.  0.5 0. ]]
'''

# 接下来对变换后的邻接矩阵应用传播规则
# 得到与相邻节点的特征均值对应的节点表征。这是因为（变换后）邻接矩阵的权重对应于相邻节点特征加权和的权重
print("D⁻¹ * A * X = \n", D**-1 * A * X)
'''
 [[ 1.  -1. ]
 [ 2.5 -2.5]
 [ 1.  -1. ]
 [ 1.  -1. ]]
'''

# 接下来得到D_hat, 是 A_hat = A + I 对应的度矩阵
D_hat = np.array(np.sum(A_hat, axis=1))[:, 0]
D_hat = np.matrix(np.diag(D_hat))
print("D_hat = \n", D_hat)
'''
 [[2. 0. 0. 0.]
 [0. 3. 0. 0.]
 [0. 0. 2. 0.]
 [0. 0. 0. 3.]]
'''


# ================= 整合 ===============

# 现在，我们将把自环和归一化技巧结合起来。
# 此外，我们还将重新介绍之前为了简化讨论而省略的有关权重和激活函数的操作。

# 添加权重
# 我们想要减小输出特征表征的维度，我们可以减小权重矩阵 W 的规模
# 但是这里故意增大了
W = np.matrix([
    [1, -1, 2],
    [-1, 1, -2]
])
print("W = \n", W)
'''
W =
 [[ 1 -1  2]
 [-1  1 -2]]
'''

print("A_hat = \n", A_hat)
'''
 [[1. 1. 0. 0.]
 [0. 1. 1. 1.]
 [0. 1. 1. 0.]
 [1. 0. 1. 1.]]
'''

print("D_hat**-1 * A_hat = \n", D_hat**-1 * A_hat)
'''
 [[0.5        0.5        0.         0.        ]
 [0.         0.33333333 0.33333333 0.33333333]
 [0.         0.5        0.5        0.        ]
 [0.33333333 0.         0.33333333 0.33333333]]
'''

print("D_hat**-1 * A_hat * X = \n", D_hat**-1 * A_hat * X)
'''
 [[ 0.5        -0.5       ]
 [ 2.         -2.        ]
 [ 1.5        -1.5       ]
 [ 1.66666667 -1.66666667]]
'''

print("D_hat**-1 * A_hat * X * W = \n", D_hat**-1 * A_hat * X * W)
'''
 [[ 1.         -1.          2.        ]
 [ 4.         -4.          8.        ]
 [ 3.         -3.          6.        ]
 [ 3.33333333 -3.33333333  6.66666667]]
'''

# 添加激活函数
print("relu(D_hat**-1 * A_hat * X * W) = \n", relu(D_hat**-1 * A_hat * X * W))
'''
 [[1.         0.         2.        ]
 [4.         0.         8.        ]
 [3.         0.         6.        ]
 [3.33333333 0.         6.66666667]]

'''


# ============ 我们将图卷积网络应用到一个真实的图上 ============

import networkx as nx
from networkx import to_numpy_matrix

zkc = nx.karate_club_graph()
print(zkc)  # Zachary's Karate Club
print(type(zkc))  # <class 'networkx.classes.graph.Graph'>

print(zkc.nodes())
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
#  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
print(type(zkc.nodes()))
# <class 'networkx.classes.reportviews.NodeView'>

order = sorted(list(zkc.nodes()))
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
#  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

A = to_numpy_matrix(zkc, nodelist=order)
print(A)
'''
[[0. 1. 1. ... 1. 0. 0.]
 [1. 0. 1. ... 0. 0. 0.]
 [1. 1. 0. ... 0. 1. 0.]
 ...
 [1. 0. 0. ... 0. 1. 1.]
 [0. 0. 1. ... 1. 0. 1.]
 [0. 0. 0. ... 1. 1. 0.]]
'''
print(A.shape[0], A.shape[1])
# 34 34

I = np.eye(zkc.number_of_nodes())  # 34 x 34
A_hat = A + I

D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

# 接下来，我们将随机初始化权重。
W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
'''
W_1 =
 [[ 6.42144245e-01 -2.83590736e-01 -8.75764693e-01  4.17843912e-01]
 [-6.60605015e-01 -4.72658496e-01  1.10796818e+00 -1.64596954e+00]
 ...
 [-4.34570333e-01 -1.17468794e+00  3.94254896e-01  2.24888554e-01]]
'''
W_2 = np.random.normal(
    loc = 0, size=(W_1.shape[1], 2))
'''
W_2 =
 [[-0.41063717  0.61026865]
 [-0.20577127 -1.79543329]
 [ 1.1148323   0.34126572]]
'''

print("W_1 = \n", W_1)
print("W_2 = \n", W_2)

# 接着，我们会堆叠 GCN 层。
# 这里，我们只使用单位矩阵作为特征表征，
# 即每个节点被表示为一个 one-hot 编码的类别变量。


def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)


H_1 = gcn_layer(A_hat, D_hat, I, W_1)

H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

output = H_2
'''
output =
 [[0.30333074 0.38689069]
 [0.34164209 0.3803171 ]
 ...
 [0.29526046 0.35807214]]
'''

print("output = \n", output)
print(output.shape[0], output.shape[1])


feature_representations = {
    node: np.array(output)[node]
    for node in zkc.nodes()}

print(feature_representations)
print(type(feature_representations))
print(feature_representations[0])

print(feature_representations[0][1])

print("len = ", len(feature_representations))

# 绘画
import matplotlib.pyplot as plt

# 原本的关系图
def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    #% matplotlib
    #notebook
    import matplotlib.pyplot as plt

    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None

    if weight_name:
        weights = [int(G[u][v][weight_name]) for u, v in edges]
        labels = nx.get_edge_attributes(G, weight_name)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nodelist1 = []
        nodelist2 = []
        for i in range(34):
            if zkc.nodes[i]['club'] == 'Mr. Hi':
                nodelist1.append(i)
            else:
                nodelist2.append(i)
        # nx.draw_networkx(G, pos, edges=edges);
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist1, node_size=300, node_color='r', alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist2, node_size=300, node_color='b', alpha=0.8)
        nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.4)


plot_graph(zkc)

# 隐层参数的图
plt.figure()
for i in range (34):
    if zkc.nodes[i]['club'] == 'Mr. Hi':
        plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,color = 'b',alpha=0.5,s = 100)
    else:
        plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,color = 'r',alpha=0.5,s = 100)


H = nx.Graph()

node_num = len(feature_representations)

nodes = list(range(node_num))  # 34 nodes

# add edges
for i in range(node_num):
    src = i
    for j in range(node_num):
        if A[i, j] != 0 and i != j:
            dest = j
            H.add_edge(src, dest)

nx.draw_networkx_edges(H, feature_representations, alpha=0.3)
plt.show()
