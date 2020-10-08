import dgl
import torch

g = dgl.DGLGraph()

g.add_nodes(5)
g.add_edges([1, 3, 4], 2)
g.ndata['x'] = torch.zeros((5, 3))
print(g.ndata)
g.nodes[2, 4].data['x'] = torch.ones((2, 3))
g.nodes[3].data['x'][0] = -1
print(g.nodes[3].data['x'])
print(g.ndata)

g.edata['edge_ft0'] = torch.zeros((3, 4))
print(g.edata)
g.edges[3, 2].data['edge_ft0'] = torch.ones(1, 4)
print(g.edata)


def send_source(edges): return {'m': edges.src['x'] * 5}


def simple_reduce(nodes): return {'x': nodes.mailbox['m'][0]}


g.register_message_func(send_source)
g.send(g.edges())
g.register_reduce_func(simple_reduce)
# g.send_and_recv(edges=g.edges(), v=2)
g.recv(v=2)
print(g.ndata)

print(g)
edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                 (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
                 (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
                 (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                 (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
                 (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                 (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                 (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                 (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                 (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                 (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                 (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                 (33, 31), (33, 32)]
# add edges two lists of nodes: src and dst
src, dst = tuple(zip(*edge_list))
print(src)
print(dst)
