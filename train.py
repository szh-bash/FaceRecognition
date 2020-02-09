import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from frModels.vggnet.vgg16 import Vgg16
from loss import ArcMarginProduct as Arcface

from init import DataReader

save_path = '/data/shenzhonghai/FaceClustering/models/Vgg16_wf_af_128_1_ep'

# set config
data = DataReader('train', 'webface')
batch_size = 128
Total = 360
learning_rate = 0.001


def get_label(output):
    # print(output.shape)
    return torch.argmax(output, dim=1)


grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def get_max_gradient(g):
    print('Gradient:', g)
    pm = torch.max(g)
    nm = torch.min(g)
    if -nm > pm:
        return nm
    else:
        return pm


if __name__ == '__main__':
    # Some Args setting
    net = Vgg16('train', 'arcface')
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1:
        devices_ids = [0, 1, 2, 3]
        net = nn.DataParallel(net, device_ids=devices_ids)
        print("Let's use %d/%d GPUs!" % (len(devices_ids), torch.cuda.device_count()))
    net.to(device)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print(net.parameters())
    arcface = Arcface(4096, data.person).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    torch.save(net.state_dict(), save_path+str(0)+'.pt')
    print('Model saved to %s' % (save_path + str(0) + '.pt'))

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("Parameter Number: %d M" % (num_params / 1e6))

    print("Training Started!")
    iterations = 0
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1, last_epoch=-1)
    for epoch in range(Total):
        data_time, train_time = 0, 0
        pred, train_x, train_y, loss = None, None, None, None

        batch_data_time = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            train_x, train_y = inputs.to(device), labels.to(device)
            dt = time.time() - batch_data_time
            data_time = data_time + dt

            batch_train_time = time.time()
            # learning_rate = adjust_lr(optimizer, epoch, learning_rate)
            # print(optimizer.state_dict()['param_groups'])
            feat = net(train_x)
            if Vgg16.loss_type == 'arcface':
                feat = arcface(feat, train_y)
            feat.register_hook(save_grad('feat_grad'))
            loss = criterion(feat, train_y)
            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            optimizer.step()    # Does the update
            scheduler.step()
            tt = time.time() - batch_train_time
            train_time = train_time + tt

            iterations += 1
            if iterations % 50 == 0:
                print('Output Max pred: %s' % torch.max(feat.gather(1, train_y.view(-1, 1))))
                print('Abs Max Gradient of net_output:', get_max_gradient(grads['feat_grad'].gather(1, train_y.view(-1, 1))))
            if iterations % 1 == 0:
                test_time = time.time()
                pred = get_label(feat)
                acc = (pred == train_y).sum().float() / train_y.size(0) * 100
                test_time = time.time() - test_time
                print('epoch: %d/%d, iters: %d, lr: %.6f, '
                      'loss: %.5f, acc: %.3f, train_time: %.4f, test_time: %.5f, data_time: %.4f' %
                      (epoch, Total, iterations, scheduler.get_lr()[0],
                       float(loss), float(acc), tt, test_time, dt))

            batch_data_time = time.time()

        epoch += 1
        if epoch % 1 == 0:
            # pred = get_label(feat)
            # acc = (pred == train_y).sum()
            print('epoch: %d/%d, loss: %.5f, train_time: %.5f, data_time: %.5f' %
                  (epoch, Total, float(loss), train_time, data_time))
        if epoch > 120 and epoch % 20 == 0:
            torch.save(net.state_dict(), save_path+str(epoch)+'.pt')
            print('Model saved to %s' % (save_path+str(epoch)+'.pt'))

    print('fydnb!')