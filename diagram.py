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
    m = 10
    n = len(seq)
    res = []
    for i in range(n-m):
        sum_ = 0
        for j in range(m):
            sum_ += seq[i+j]
        res.append(sum_/m)
    return res


log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_bs-128_lr-8|12k|15k.log'
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
        loc = re.search(r'loss: [\d]*\.[\d]*', st).span()
        loss.append(float(st[loc[0]+6:loc[1]]))
        if iters % 104 == 0:
            continue
        loc = re.search(r'acc: [\d]*', st).span()
        acc.append(int(st[loc[0]+5:loc[1]]))
loss = smooth(loss)
acc = smooth(acc)

iters = len(acc)
x = np.linspace(0, iters, iters)
# plt.plot(x, loss, label='loss')
# plt.ylim(0., 9.)
plt.plot(x, acc, label='acc')
plt.ylim(0., 128.)
plt.xlabel('iterations')
plt.ylabel('Value')
# plt.title("Loss Diagram")
plt.title("Acc Diagram")
plt.legend()

plt.show()
