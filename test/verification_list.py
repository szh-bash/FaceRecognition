import os
import sys
import numpy as np
from config import verificationPath, dataPath


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def build_same():
    for i in range(m):
        w = rng.randint(pss) % pss
        while ed[cls[w]] == st[cls[w]]:
            w = rng.randint(pss) % pss
        tot = ed[cls[w]] - st[cls[w]]
        u = rng.randint(tot) % tot + st[cls[w]]
        v = rng.randint(tot) % tot + st[cls[w]]
        x = imgs[u]
        y = imgs[v]
        print(x[0], x[1], y[0], y[1], int(x[0] == y[0]))
        pass


def build_diff():
    for i in range(m):
        x = imgs[rng.randint(n) % n]
        y = imgs[rng.randint(n) % n]
        while x[0] == y[0]:
            x = imgs[rng.randint(n) % n]
            y = imgs[rng.randint(n) % n]
        print(x[0], x[1], y[0], y[1], int(x[0] == y[0]))


if __name__ == "__main__":
    data_name = 'pie'
    path = dataPath[data_name]
    path_dir = os.listdir(path)

    imgs = []
    cls = []
    cnt = 0
    st = {}
    ed = {}
    for allDir in path_dir:
        child = os.path.join('%s/%s' % (path, allDir))
        child_dir = os.listdir(child)
        cls.append(allDir)
        st[allDir] = cnt
        for allSon in child_dir:
            allSon = allSon.split('.')
            imgs.append([allDir, allSon['pie' not in data_name]])
            cnt += 1
        ed[allDir] = cnt
    imgs = np.array(imgs)
    print(imgs.shape)
    n = imgs.shape[0]
    m = 3000
    pss = len(cls)

    sys.stdout = Logger(verificationPath[data_name], sys.stdout)
    np.random.seed(233)
    rng = np.random
    print(m+m)
    build_same()
    build_diff()
