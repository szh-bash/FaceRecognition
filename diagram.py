import re
import numpy as np
import matplotlib.pyplot as plt


def smooth(seq):
    m = 5
    n = len(seq)
    res = []
    for i in range(n-m):
        sum_ = 0
        for j in range(m):
            sum_ += seq[i+j]
        res.append(sum_/m)
    return res


# log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_bs-128_lr-4|16k|19k.log'
# log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_af_128_3.log'
# log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_af_128_3|20k.log'
# log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_lfw_sm_128_3.log'
# log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_wf_sm_128_2.log'
log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_lfw_af-0.2_flip_256_2.log'
log_path = '/data/shenzhonghai/FaceClustering/logs/train_log_Vgg16_lfw_af-2_flip_256_2.log'
acc = []
loss = []
iters = 0
with open(log_path, 'r') as f:
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
        acc.append(float(st[loc[0]+5:loc[1]]))
loss = smooth(loss)
acc = smooth(acc)
print(log_path)

iters = len(acc)
x = np.linspace(0, iters, iters)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, loss, label='loss', color='r')
ax2.plot(x, acc, label='train_acc', color='b')
ax1.set_xlim(0, iters)
ax1.set_ylim(0.,24)
ax2.set_ylim(0., 100.)
ax1.set_ylabel('loss')
ax2.set_ylabel('train_acc')
plt.xlabel('iterations')
plt.title(log_path)
fig.legend(bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes)

plt.show()
