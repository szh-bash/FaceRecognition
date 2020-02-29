import os
import re
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
import progressbar as pb

sys.path.append("..")
from init import DataReader
from model.vggnet.vgg16 import Vgg16

from config import featPath, modelPath, dp


# check = torch.load(modelPath)
# print(check)
# exit(0)


def save_feat(ft, name_list, lim, path):
    ft = ft.cpu()
    if not os.path.exists(path):
        os.mkdir(path)
    for dx in range(lim):
        filepath, filename = os.path.split(name_list[dx])
        loc = re.search(r'[\d]+', filename).span()
        name = filename[0:loc[0]-1]
        idx = int(filename[loc[0]:loc[1]])
        if not os.path.exists(path+name):
            os.mkdir(path+name)
        ftx = ft[dx].data.numpy()
        ftx = ftx.tolist()
        pt = pd.DataFrame(data=ftx)
        pt.to_csv(path+name+'/'+str(idx), mode='w', index=None, header=None)


# load data
batch_size = 1
data = DataReader('test', 'mtLfw')
data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, pin_memory=True)

# load model
# model_path = '/data/shenzhonghai/FaceClustering/models/Vgg16_bs-128_lr-4|16k|19k_ep200.pt'
# model_path = '/data/shenzhonghai/FaceClustering/models/Vgg16_af_128_3|20k_ep210.pt'
# model_path = '/data/shenzhonghai/FaceClustering/models/Vgg16_lfw_af-1_256_2_ep100.pt'
# model_path = '/data/shenzhonghai/FaceClustering/models/Vgg16_wf_af-1_256_lr1e3_2|60k_ep35.pt'
device = torch.device('cuda:0')
model = Vgg16().cuda()
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelPath).items()})
model.eval()  # DropOut/BN
# print('epoch: %d, iter: %d, loss: %.5f, train_acc: %.5f' %
#       (checkpoint['epoch'], checkpoint['iter'], checkpoint['loss'], checkpoint['acc']))

# get feat
print('Calculating Feature Map...')
ids = 0
Total = (data.len - 1) / batch_size + 1
widgets = [' ', pb.Percentage(),
           ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
           ' ', pb.Timer(),
           ' ', pb.ETA(),
           ' ', pb.FileTransferSpeed()]
pgb = pb.ProgressBar(widgets=widgets, maxval=Total).start()
for i, (inputs, labels, names) in enumerate(data_loader):
    feat = model(inputs.to(device))
    save_feat(feat, names, labels.size(0), featPath)
    pgb.update(i)
pgb.finish()
print('Feature Map saved to \'%s\' successfully!' % featPath[:-1])
