import os
import sys
import numpy as np


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
    path = '/data/shenzhonghai/Face_Recognition_Data/grimace/'
    # path = '/data/shenzhonghai/Face_Recognition_Data/faces96-112x112/'
    # path = '/data/shenzhonghai/PIE_dataset/test-112x112/'
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
            imgs.append([allDir, allSon['PIE' not in path]])
            cnt += 1
        ed[allDir] = cnt
    imgs = np.array(imgs)
    print(imgs.shape)
    n = imgs.shape[0]
    m = 3000
    pss = len(cls)

    sys.stdout = Logger('/data/shenzhonghai/Face_Recognition_Data/grimace-pairs.txt', sys.stdout)
    # sys.stdout = Logger('/data/shenzhonghai/Face_Recognition_Data/faces96-pairs.txt', sys.stdout)
    # sys.stdout = Logger('/data/shenzhonghai/PIE_dataset/pairs.txt', sys.stdout)
    np.random.seed(233)
    rng = np.random
    print(m+m)
    build_same()
    build_diff()
