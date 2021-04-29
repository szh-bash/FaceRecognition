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


# path = '/data/shenzhonghai/Face_Recognition_Data/faces95-112x112/'
path = '/data/shenzhonghai/PIE_dataset/test-112x112/'
path_dir = os.listdir(path)
imgs = []
for allDir in path_dir:
    child = os.path.join('%s/%s' % (path, allDir))
    child_dir = os.listdir(child)
    for allSon in child_dir:
        allSon = allSon.split('.')
        # imgs.append([allSon[0], allSon[1]])
        imgs.append([allDir, allSon[0]])
imgs = np.array(imgs)
# print(imgs)
print(imgs.shape)
n = imgs.shape[0]
m = 3000

# sys.stdout = Logger('/data/shenzhonghai/Face_Recognition_Data/faces95-pairs.txt', sys.stdout)
sys.stdout = Logger('/data/shenzhonghai/PIE_dataset/pairs.txt', sys.stdout)
np.random.seed(233)
rng = np.random
print(m)
for i in range(m):
    x = imgs[rng.randint(n) % n]
    y = imgs[rng.randint(n) % n]
    print(x[0], x[1], y[0], y[1], int(x[0] == y[0]))
