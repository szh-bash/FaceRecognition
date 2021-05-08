import re
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from r50Feat import get
from init import DataReader
from config import modelPath, server, verificationPath
import socket
import time
from r50Feat import get
from config import modelPath, server, verificationPath, dataPath


def get_distance(feat0, feat1):
    return np.sum(np.multiply(feat0, feat1)) / (np.linalg.norm(feat0) * np.linalg.norm(feat1))


def identify(cls):
    ma = {}
    _pos = 0
    _cnt = 0
    for sp in sps_test[cls]:
        lst = [get_distance(store_test[sp], store_base[a]) for a in store_base]
        ma[sp] = max(lst)
        for b in store_test:
            if cls not in b:
                ma[sp] = max(ma[sp], get_distance(store_test[sp], store_test[b]))
    for sp in sps_test[cls]:
        for sp2 in sps_test[cls]:
            if sp != sp2:
                _pos += get_distance(store_test[sp], store_test[sp2]) > ma[sp]
                _cnt += 1

    print("Rank 1 Acc@1e-5: %.3f (%s)" % (_pos/_cnt*100, cls))
    return _pos, _cnt


if __name__ == '__main__':
    test_data = 'faces95C'
    filepath = dataPath[test_data]
    print(filepath)
    data_base = DataReader('test', 'RetinaLfwCenter')
    data_test = DataReader('test', test_data)
    store_base, feats_base, sps_base = get(modelPath, data_base)
    store_test, feats_test, sps_test = get(modelPath, data_test)
    pos_sum = 0
    cnt_sum = 0
    for x in sps_test:
        pos, cnt = identify(x)
        pos_sum += pos
        cnt_sum += cnt
    print("Rank 1 Acc@1e-5: %.3f" % (pos_sum/cnt_sum*100))

