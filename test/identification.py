import sys
import numpy as np
sys.path.append("..")
from init import DataReader
from r50Feat import get
from config import modelPath, dataPath


def get_distance(feat0, feat1):
    return np.sum(np.multiply(feat0, feat1)) / (np.linalg.norm(feat0) * np.linalg.norm(feat1))


def identify(cls):
    if len(sps_test[cls]) < 2:
        return 0, 0
    ma = {}
    _pos = 0
    _cnt = 0
    lst = []
    for sp in sps_test[cls]:
        for a in store_base:
            if cls not in a.split('/')[0]:
                lst.append(get_distance(store_test[sp], store_base[a]))
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
    test_data = 'RetinaLfwCenter'
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
    print(modelPath)
    print(test_data)

