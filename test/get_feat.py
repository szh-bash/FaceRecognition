import os
import re
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
import progressbar as pb

sys.path.append("..")
from init import DataReader
from frModels.vggnet.vgg16 import Vgg16


def save_feat(ft, name_list, lim, path):
    ft = ft.cpu()
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


# load model
model_path = '/data/shenzhonghai/FaceClustering/models/Vgg16_bs-128_lr-4|16k|19k_ep200.pt'
device = torch.device('cuda:0')
model = Vgg16().cuda()
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
model.eval()  # DropOut/BN
print(model)

# load data
feat_path = '/data/shenzhonghai/lfw/lfw-feat/'
batch_size = 32
data = DataReader('test')
data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, pin_memory=True)
print('Calculating Feature Map...')
ids = 0
Total = (13233 - 1) / batch_size + 1
widgets = [' ', pb.Percentage(),
           ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
           ' ', pb.Timer(),
           ' ', pb.ETA(),
           ' ', pb.FileTransferSpeed()]
pgb = pb.ProgressBar(widgets=widgets, maxval=Total).start()
for i, (inputs, labels, names) in enumerate(data_loader):
    feat = model(inputs.to(device))
    save_feat(feat, names, len(inputs), feat_path)
    pgb.update(i)
pgb.finish()
print('Feature Map saved to \'%s\' successfully!' % feat_path[:-1])
