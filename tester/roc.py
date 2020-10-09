import re
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from tester.getFeat import get
from init import DataReader
from config import modelPath, pairsTxtPath, server
import socket
import time


def get_distance(path0, path1, **__store):
    feat0, feat1 = __store[path0], __store[path1]
    # return np.argmax(feat0) == np.argmax(feat1)
    # return -np.linalg.norm(feat0-feat1)
    return np.sum(np.multiply(feat0, feat1)) / (np.linalg.norm(feat0) * np.linalg.norm(feat1))


def get_img_pairs_list(**_store):
    file = open(pairsTxtPath)
    name_pattern = re.compile(r'[a-z|_\-]+', re.I)
    id_pattern = re.compile(r'[0-9]+')
    st = file.readline()
    times, batches = id_pattern.findall(st)[0:2]
    times, batches = int(times), int(batches)
    total = times * batches * 2
    res_dist = []
    res_gt = []
    for i in range(total):
        st = file.readline()
        flag = (i//batches) & 1
        names = name_pattern.findall(st)
        ids = id_pattern.findall(st)
        res_dist.append(get_distance(names[0]+'/'+ids[0], names[flag]+'/'+ids[1], **_store))
        res_gt.append(flag ^ 1)
    file.close()
    return total, res_dist, res_gt


def get_acc(threshold_list, cases, dist, ground_truth):
    res = []
    for threshold in threshold_list:
        res.append(((dist > threshold) == ground_truth).sum())
    res = np.array(res) / cases * 100
    return res


def global_calc(_index, _ground_truth, __test_total):
    _roc = 0
    pos = 0
    neg = 0
    _max_test_acc = 0
    __true_ratio = []
    for idx in _index:
        if _ground_truth[idx]:
            pos += 1
            if len(__true_ratio) > 0:
                __true_ratio[-1] += 1 / __test_total * 2
        else:
            neg += 1
            if len(__true_ratio) > 0:
                _roc += pos * pow(1 / __test_total * 2, 2)
            __true_ratio.append(pos / __test_total * 2)
        _max_test_acc = max(_max_test_acc, (pos+__test_total//2-neg)/__test_total)
    __true_ratio[-1] = 1.0
    return __true_ratio, _max_test_acc*100, _roc


def cross_acc(_dist, _ground_truth):
    res = 0
    for i in range(10):
        p = _dist[i*600:(i+1)*600]
        gt = _ground_truth[i*600:(i+1)*600]
        _index = np.argsort(p)[::-1]
        pos = 0
        neg = 0
        ma = 0
        threshold = 0
        for idx in _index:
            if gt[idx]:
                pos += 1
            else:
                neg += 1
            if pos - neg >= ma:
                ma = pos - neg
                threshold = p[idx]
        res += ((_dist > threshold) == _ground_truth).sum()
    return res/10


def calc(filepath):
    store, feats = get(filepath, data)
    _test_total, dist, ground_truth = get_img_pairs_list(**store)
    dist = np.array(dist)
    index = np.argsort(dist)[::-1]
    ground_truth = np.array(ground_truth)

    # Approximate test_acc
    _test_acc = get_acc(thresholds, _test_total, dist, ground_truth)
    print('Max test_acc: %.3f (threshold=%.5f)' % (_test_acc.max(), thresholds[_test_acc.argmax()]))

    _cross_validation = cross_acc(dist, ground_truth) / _test_total * 100

    # Roc
    _true_ratio, max_test_acc, roc = global_calc(index, ground_truth, _test_total)
    eer = np.abs(1 - np.array(_true_ratio) - np.linspace(0, 1.0, (_test_total // 2)))
    print('Global Test Accuracy: %.3f ' % max_test_acc)
    print('Cross-validation Test Accuracy: %.3f' % _cross_validation)
    print('@FAR = 0.00100: TAR = %.5f' % _true_ratio[2])
    print('@FAR = 0.01000: TAR = %.5f' % _true_ratio[29])
    print('@FAR = 0.02000: TAR = %.5f' % _true_ratio[58])
    print('EER: %.5f' % _true_ratio[np.array(eer).argmin()])
    print('AUC: %.5f' % roc)

    # lfw-test

    return _test_acc, _test_total, _true_ratio


def link_handler(link):
    print(time.strftime("%Y-%m-%d %H:%M:%S Test server activated....", time.localtime()))
    filepath = link.recv(1024).decode()
    if filepath == 'exit':
        print("Train End....")
        return True
    print(filepath)
    calc(filepath)
    link.close()
    return False


def test_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', server))
    s.listen(5)
    print(time.strftime("%Y-%m-%d %H:%M:%S Test server Online....", time.localtime()))
    while True:
        cnn, addr = s.accept()
        if link_handler(cnn):
            exit(0)


if __name__ == '__main__':
    data = DataReader('test', 'MegaLfw112')
    length = 10000
    thresholds_left, thresholds_right = -0.0, 1.0
    thresholds = np.linspace(thresholds_left, thresholds_right, length)
    test_server()
    test_acc, test_total, true_ratio = calc(modelPath)

    # plotting test_acc
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(thresholds, test_acc, label='test_acc', color='b')
    ax2.plot(np.linspace(thresholds_left, thresholds_right, test_total//2), true_ratio, label='roc', color='r')
    ax1.set_xlim(thresholds_left, thresholds_right)
    ax1.set_ylim(45, 100)
    ax2.set_ylim(0., 1.)
    ax1.set_ylabel('test_acc')
    ax2.set_ylabel('roc')
    plt.xlabel('thresholds')
    plt.title(modelPath.split('/')[-1])
    fig.legend(bbox_to_anchor=(0.6, 1.), bbox_transform=ax1.transAxes)
    plt.show()
    print(modelPath)
