import re
import numpy as np
import matplotlib.pyplot as plt

# st = 'epoch: 159/160, iters: 16640, lr: 0.000001, loss: 0.37112, acc: 46, train_time: 0.0519, test_time: 0.00015, ' \
#      'data_time: 0.0029 '
# print(re.search('z', st) is None)
# print(re.search(r'loss: [\d]*\.[\d]*, acc', st).span())
# loc = re.search(r'loss: [\d]*\.[\d]*, acc', st).span()
# print(float(st[loc[0]+6:loc[1]-5]))
# exit(0)


def smooth(seq):
    m = 20
    n = len(seq)
    res = []
    for i in range(n-m):
        sum_ = 0
        for j in range(m):
            sum_ += seq[i+j]
        res.append(sum_/m)
    return res


# log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_bs-128_lr-8|12k|15k.log'
log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_bs-128_lr-400.log'
acc = []
loss = []
iters = 0
with open(log_path, 'r') as f:
    # print(f.read())
    # exit(0)
    for st in f.readlines():
        if re.search('iters', st) is None:
            continue
        iters += 1
        if iters % 104 == 0:
            continue
        loc = re.search(r'loss: [\d]*\.[\d]*', st).span()
        loss.append(float(st[loc[0]+6:loc[1]]))
        # loc = re.search(r'acc: [\d]*', st).span()
        # acc.append(float(st[loc[0]+5:loc[1]])*100/128)
        loc = re.search(r'acc: [\d]*\.[\d]', st).span()
        acc.append(float(st[loc[0]+5:loc[1]])*100)
loss = smooth(loss)
acc = smooth(acc)

iters = len(acc)
x = np.linspace(0, iters, iters)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, loss, label='loss', color='r')
ax2.plot(x, acc, label='acc', color='b')
ax1.set_xlim(0000,)
ax1.set_ylim(0., 8.)
# ax2.set_ylim(40., 100.)
ax1.set_ylabel('loss')
ax2.set_ylabel('acc')
plt.xlabel('iterations')
plt.title("Loss/Acc Diagram")
fig.legend(bbox_to_anchor=(0.3, 1), bbox_transform=ax1.transAxes)

plt.show()
