import re
import sys
import numpy as np
import matplotlib.pyplot as plt
# import progressbar as pb
sys.path.append("..")
from config import modelPath, pairsTxtPath
from get_feat import store

dist = []
ground_truth = []
# widgets = ['Testing: ', pb.Percentage(),
#            ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
#            ' ', pb.Timer(),
#            ' ', pb.ETA(),
#            ' ', pb.FileTransferSpeed()]


def get_distance(path0, path1):
    # feat0, feat1 = [], []
    # file = open(path0)
    # for val in file:
    #     feat0.append(float(val))
    # file.close()
    # file = open(path1)
    # for val in file:
    #     feat1.append(float(val))
    # file.close()
    # feat0, feat1 = np.array(feat0), np.array(feat1)
    feat0, feat1 = store[path0], store[path1]
    # return np.argmax(feat0) == np.argmax(feat1)
    # return -np.linalg.norm(feat0-feat1)
    return np.sum(np.multiply(feat0, feat1)) / (np.linalg.norm(feat0) * np.linalg.norm(feat1))


def get_img_pairs_list():
    file = open(pairsTxtPath)
    name_pattern = re.compile(r'[a-z|_|\-]+', re.I)
    id_pattern = re.compile(r'[0-9]+')
    st = file.readline()
    times, batches = id_pattern.findall(st)[0:2]
    times, batches = int(times), int(batches)
    # print(times, batches)
    total = times * batches * 2
    # pgb = pb.ProgressBar(widgets=widgets, maxval=total).start()
    for i in range(total):
        st = file.readline()
        flag = (i//batches) & 1
        names = name_pattern.findall(st)
        ids = id_pattern.findall(st)
        dist.append(get_distance(names[0]+'/'+ids[0], names[flag]+'/'+ids[1]))
        ground_truth.append(flag ^ 1)
        # pgb.update(i)
    # pgb.finish()
    file.close()
    # print(dist, '\n', ground_truth)
    print(dist[:5])
    print(dist[-5:])
    return total


def get_acc(threshold_list, cases):
    res = []
    for threshold in threshold_list:
        res.append(((dist > threshold) == ground_truth).sum())
    res = np.array(res) / cases * 100
    return res


test_total = get_img_pairs_list()
dist = np.array(dist)
ground_truth = np.array(ground_truth)

# Figure out test_acc
length = 10000
thresholds_left, thresholds_right = -0.0, 1.0
thresholds = np.linspace(thresholds_left, thresholds_right, length)
test_acc = get_acc(thresholds, test_total)
print('Max test_acc: %.3f (threshold=%.5f)' % (test_acc.max(), thresholds[test_acc.argmax()]))
# print((dist == ground_truth).sum()/test_total*100)

# Roc
index = np.argsort(dist)[::-1]
roc = 0
count = 0
true_ratio = []
for idx in index:
    if ground_truth[idx]:
        count += 1
        if len(true_ratio) > 0:
            true_ratio[-1] += 1 / test_total * 2
    else:
        if len(true_ratio) > 0:
            roc += count * pow(1 / test_total * 2, 2)
        true_ratio.append(count / test_total * 2)
true_ratio[-1] = 1.0
# print(true_ratio)
eer = np.abs(1 - np.array(true_ratio) - np.linspace(0, 1.0, (test_total/2)))
print('EER: %.5f' % true_ratio[np.argmin(eer)])
print('AUC: %.5f' % roc)
print('@FAR = 0.00100: TAR = %.5f' % true_ratio[2])
print('@FAR = 0.01000: TAR = %.5f' % true_ratio[29])
print('@FAR = 0.02000: TAR = %.5f' % true_ratio[58])

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
