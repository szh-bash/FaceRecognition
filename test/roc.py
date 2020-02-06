import re
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb


pairs_txt_path = '/data/shenzhonghai/lfw/pairs.txt'
feat_path = '/data/shenzhonghai/lfw/lfw-feat/'
dist = []
ground_truth = []
widgets = ['Testing: ', pb.Percentage(),
           ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
           ' ', pb.Timer(),
           ' ', pb.ETA(),
           ' ', pb.FileTransferSpeed()]


def get_distance(path0, path1):
    # feat0 = pd.read_csv(path0, index_col=None, header=None)
    # feat1 = pd.read_csv(path1, index_col=None, header=None)
    feat0, feat1 = [], []
    file = open(path0)
    for val in file:
        feat0.append(float(val))
    file.close()
    file = open(path1)
    for val in file:
        feat1.append(float(val))
    file.close()
    feat0, feat1 = np.array(feat0), np.array(feat1)
    # return np.linalg.norm(feat0-feat1)
    return np.sum(np.multiply(feat0, feat1)) / (np.linalg.norm(feat0) * np.linalg.norm(feat1))


def get_img_pairs_list():
    file = open(pairs_txt_path)
    name_pattern = re.compile(r'[a-z|_|\-]+', re.I)
    id_pattern = re.compile(r'[0-9]+')
    st = file.readline()
    times, batches = id_pattern.findall(st)[0:2]
    times, batches = int(times), int(batches)
    # print(times, batches)
    total = times * batches
    pgb = pb.ProgressBar(widgets=widgets, maxval=total).start()
    for i in range(total):
        st = file.readline()
        flag = (i//batches) & 1
        names = name_pattern.findall(st)
        ids = id_pattern.findall(st)
        dist.append(get_distance(feat_path+names[0]+'/'+ids[0], feat_path+names[flag]+'/'+ids[1]))
        ground_truth.append(flag ^ 1)
        pgb.update(i)
    pgb.finish()
    file.close()
    print(dist, '\n', ground_truth)
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
# test_acc = get_acc([2333, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 12222, 15000, 23333], test_total)
test_acc = get_acc([0.95, 0.96, 0.9625, 0.965, 0.96625, 0.9675, 0.97, 0.98], test_total)
# test_acc = get_acc([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], test_total)
print(test_acc)
exit(0)


def plot_roc(predStrengths, classLabels):
    cur = (0.0, 0.0)
    numPosClass = np.sum(np.array(classLabels) == 1.0)
    yStep = 1.0/numPosClass
    xStep = 1.0/(len(classLabels)-numPosClass)
    print(np.array(predStrengths.flatten()))
    sortedIndicies = np.argsort(-np.array(predStrengths.flatten()))
    print(sortedIndicies)
    fig = plt.figure()
    fig.clf()
    ySum = 0.0
    ax = plt.subplot(111)
    for index in sortedIndicies:
        if classLabels[index] == 1.0:
            delY = yStep; delX=0
        else:
            delY = 0; delX = xStep
            ySum += cur[1]
        ax.plot([cur[0], cur[0]+delX], [cur[1], cur[1]+delY], c='b')
        cur = (cur[0]+delX, cur[1]+delY)
        print(cur)
    ax.plot([0, 1], [0, 1], 'b--')
    ax.axis([0, 1, 0, 1])
    plt.xlabel('False Positve Rate')
    plt.ylabel('True Postive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area under the curve is:', ySum*xStep)


plot_roc(100, [0, 1])
