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


def get_distance(path0, path1, **__store):
    feat0, feat1 = __store[path0], __store[path1]
    # return np.argmax(feat0) == np.argmax(feat1)
    # return -np.linalg.norm(feat0-feat1)
    return np.sum(np.multiply(feat0, feat1)) / (np.linalg.norm(feat0) * np.linalg.norm(feat1))


def get_img_pairs_list(**_store):
    file = open(verificationPath[test_data])
    name_pattern = re.compile(r'[a-z|_\-]+', re.I)
    id_pattern = re.compile(r'[0-9]+')
    st = file.readline()
    times, batches = id_pattern.findall(st)[0:2]
    times, batches = int(times), int(batches)
    total = times * batches * 2
    res_dist = []
    res_gt = []
    lst = []
    for i in range(total):
        st = file.readline()
        lst.append(st)
        flag = (i//batches) & 1
        names = name_pattern.findall(st)
        ids = id_pattern.findall(st)
        res_dist.append(get_distance(names[0]+'/'+ids[0], names[flag]+'/'+ids[1], **_store))
        res_gt.append(flag ^ 1)
    file.close()
    return total, res_dist, res_gt, lst


def verification(**_store):
    file = open(verificationPath[test_data])
    st = file.readline()
    total = int(st)
    res_dist = []
    res_gt = []
    pos = 0
    neg = 0
    lst = []
    for i in range(total):
        st = file.readline()[:-1].split(' ')
        res_dist.append(get_distance(st[0]+'/'+st[1], st[2]+'/'+st[3], **_store))
        res_gt.append(int(st[4]))
        pos += int(st[4])
        neg += 1 ^ int(st[4])
        lst.append(st)
    file.close()
    return total, res_dist, res_gt, pos, neg, lst


def get_acc(cases, dist, ground_truth):
    res = []
    for x in thresholds:
        res.append(((dist > x) == ground_truth).sum())
    res = np.array(res) / cases * 100
    return res


def global_calc(index, ground_truth, __test_total, __pos, __neg):
    roc = 0
    pos = 0
    neg = 0
    max_test_acc = 0
    __true_ratio = [0]
    for idx in index:
        if ground_truth[idx]:
            pos += 1
            __true_ratio[-1] += 1 / __pos
        else:
            neg += 1
            roc += pos * pow(1 / __pos, 2)
            __true_ratio.append(pos / __pos)
        max_test_acc = max(max_test_acc, (pos+__neg-neg)/__test_total)
    return __true_ratio, max_test_acc*100, roc


def cross_acc(dist, ground_truth):
    res = []
    for i in range(10):
        p = dist[i*600:(i+1)*600].copy()
        gt = ground_truth[i*600:(i+1)*600].copy()
        res.append(max((((dist > x) == ground_truth).sum() - ((p > x) == gt).sum()) for x in thresholds)/5400*100)
    ave = np.mean(res)
    std = np.std(res)
    return ave, std


def print_wrong_sample(dist, ground_truth, threshold, lst):
    res = (dist > threshold) != ground_truth
    for i in range(len(res)):
        if not res[i]:
            continue
        print(lst[i])


def calc(filepath):
    store, feats, sps = get(filepath, data)
    md = 'Lfw' in data.data_name
    if md:
        _test_total, dist, ground_truth, lst = get_img_pairs_list(**store)
        pos = 3000
        neg = 3000
    else:
        _test_total, dist, ground_truth, pos, neg, lst = verification(**store)
        print(pos, neg)

    dist = np.array(dist)
    index = np.argsort(dist)[::-1]
    ground_truth = np.array(ground_truth)

    # Approximate test_acc
    _test_acc = get_acc(_test_total, dist, ground_truth)
    threshold = thresholds[_test_acc.argmax()]
    print('Max test_acc: %.3f (threshold=%.5f)' % (_test_acc.max(), threshold))

    # Specific
    _true_ratio, max_test_acc, roc = global_calc(index, ground_truth, _test_total, pos, neg)
    eer = np.abs(1 - np.array(_true_ratio) - np.linspace(0, 1.0, len(_true_ratio)))
    print('Global Test Accuracy: %.3f ' % max_test_acc)

    if md:
        cross_val, cross_std = cross_acc(dist, ground_truth)
        print('Cross-validation Test Accuracy: %.3f%c%.3f' % (cross_val, chr(177), cross_std))
    print('@FAR = 0.00001: TAR = %.5f' % _true_ratio[int(neg*0.00001)])
    print('@FAR = 0.00010: TAR = %.5f' % _true_ratio[int(neg*0.00010)])
    print('@FAR = 0.00100: TAR = %.5f' % _true_ratio[int(neg*0.00100)])
    print('@FAR = 0.01000: TAR = %.5f' % _true_ratio[int(neg*0.01000)])
    print('EER: %.5f' % _true_ratio[np.array(eer).argmin()])
    print('AUC: %.5f' % roc)

    return _test_acc, _test_total, _true_ratio, dist, ground_truth, threshold, lst


def link_handler(link):
    filepath = link.recv(1024).decode()
    if filepath == 'exit':
        print("Train End....")
        return True
    print(time.strftime("%Y-%m-%d %H:%M:%S Test server activated....", time.localtime()))
    print("Model path: " + filepath)
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
    test_data = 'RetinaLfwCenter'
    data = DataReader('test', test_data)
    length = 10000
    thresholds_left, thresholds_right = -0.0, 1.0
    thresholds = np.linspace(thresholds_left, thresholds_right, length)
    # test_server()
    test_acc, test_total, true_ratio, _dist, _ground_truth, _threshold, _lst = calc(modelPath)

    # plotting test_acc
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(thresholds, test_acc, label='test_acc', color='b')
    ax2.plot(np.linspace(thresholds_left, thresholds_right, len(true_ratio)), true_ratio, label='roc', color='r')
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
    print(test_data)
    # print_wrong_sample(_dist, _ground_truth, _threshold, _lst)
