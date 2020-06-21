import torch
import torch.nn.functional as F

import numpy as np

import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from model.gcn.gcn import GCN
from utils.Graph import build_karate_club_graph

# from config import
from init import DataReader

import warnings
warnings.filterwarnings('ignore')

data = DataReader('feat', 'lfw')
data.len = data.person = 34
net = GCN(data.len, 40, data.person)
print(net)
G = build_karate_club_graph()

inputs = data.feat[:34]
print(inputs.shape)
labeled_nodes = torch.tensor(data.idx)  # only the instructor and the president nodes are labeled
labels = torch.tensor(np.linspace(0, data.person, data.person, dtype=int))
print(labels[:5])

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []

for epoch in range(40):
    logits = net(G, inputs)
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)

    # compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


exit(0)


def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(data.len):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    # ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors, with_labels=True, node_size=300, ax=ax)


plt.style.use('seaborn-pastel')
nx_G = G.to_networkx().to_undirected()
fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
draw(39)  # draw the prediction of the first epoch

ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=100)
ani.save('figure.gif', writer='imagemagick')
plt.show()
