import os
import re
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from progressbar import *

sys.path.append("..")
from init import DataReader
from frModels.vggnet.vgg16 import Vgg16


def save_feat(ft, index, lim, path):
    ft = ft.cpu()
    for dx in range(lim):
        filepath, filename = os.path.split(data.name[index+dx])
        loc = re.search(r'[\d]', filename).span()
        name = filename[0:loc[0]-1]
        idx = int(filename[loc[0]:loc[1]])
        if not os.path.exists(path+name):
            os.mkdir(path+name)
        ftx = ft[dx].data.numpy()
        ftx = ftx.tolist()
        pt = pd.DataFrame(data=ftx)
        pt.to_csv(path+name+'/'+str(idx), mode='w', index=None, header=None)
        # print(path+name+'/'+str(idx), ftx)


# load model
model_path = '/data/shenzhonghai/FaceClustering/models/Vgg16_bs-128_lr-4|16k|19k_ep200.pt'
device = torch.device('cuda:0')
model = Vgg16().cuda()
# check_point = torch.load(model_path)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6])
# model.load_state_dict(check_point)
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
model.eval()  # DropOut/BN
print(model)

# load data
feat_path = '/data/shenzhonghai/lfw/lfw-feat/'
batch_size = 16
data = DataReader()
data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, pin_memory=True)
print('Calculating Feature Map...')
ids = 0
Total = (13233 - 1) / batch_size + 1
widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
pgb = ProgressBar(widgets=widgets, maxval=10 * Total).start()
for i, (inputs, labels) in enumerate(data_loader):
    feat = model(inputs.to(device))
    # print("img %d done! %s" % (labels[0], data.name[ids]))
    save_feat(feat, ids, len(inputs), feat_path)
    ids += len(inputs)
    pgb.update(i * 10 + 1)
pgb.finish()
print('Feature Map saved to \'%s\' successfully!' % feat_path[:-1])
exit(0)


def get_img_pairs_list(pairs_txt_path, img_path):
    file = open(pairs_txt_path)
    img_pairs_list, labels = [], []


def plotROC(predStrengths, classLabels):
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


plotROC(100, [0,1])
