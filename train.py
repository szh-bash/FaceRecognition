import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from frModels.vggnet.vgg16 import Vgg16
# from utils.DataHandler import Augment, DataPipe

from init import DataReader

save_path = "/home/shenzhonghai/FaceClustering/models/softmax_"

# set config
data = DataReader()
batch_size = 256
Total = 250
learning_rate = 0.01
device = torch.device("cuda:1")


def get_label(output):
    return torch.argmax(output, axis=1).reshape(-1)


def adjust_lr(opt, ep, lr):
    if ep == 30:
        lr /= 10
    elif ep == 90:
        lr /= 10
    elif ep == 210:
        lr /= 10
    else:
        return
    for pg in opt.param_groups:
        pg['lr'] = lr


if __name__ == '__main__':
    # Some Args setting
    net = Vgg16()
    if torch.cuda.device_count() > 1:
        print("Let's use %d/%d GPUs!" % (4, torch.cuda.device_count()))
        # net = nn.DataParallel(net, device_ids=[1, 2, 3, 4, 5, 6, 7, 8])
        net = nn.DataParallel(net, device_ids=[1, 2, 3, 4])
    net.to(device)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    torch.save(net.state_dict(), save_path+str(0)+'.pt')
    print('Model saved to %s' % (save_path + str(0) + '.pt'))

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print("Parameter Number: %d M" % (num_params / 1e6))

    print("Training Started!")
    for epoch in range(Total):
        data_time, train_time = 0, 0
        pred, train_x, train_y, loss = None, None, None, None

        batch_data_time = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            # inputs, labels = Variable(inputs), Variable(labels)
            train_x, train_y = inputs.to(device), labels.to(device)
            # print(train_y)
            # train_x, train_y = Variable(train_x), Variable(train_y)
            data_time = data_time + time.time() - batch_data_time

            batch_train_time = time.time()
            optimizer.zero_grad()   # zero the gradient buffers
            adjust_lr(optimizer, epoch, learning_rate)
            pred = net(train_x)
            loss = criterion(pred, train_y)
            loss.backward()
            optimizer.step()    # Does the update

            train_time = train_time + time.time() - batch_train_time
            batch_data_time = time.time()

        if epoch % 1 == 0:
            pred = get_label(pred)
            acc = (pred == train_y).sum()
            print('epoch: %d/%d, loss: %.5f, acc: %d, train_time: %.5f, data_time: %.5f' %
                  (epoch, Total, loss, acc, train_time, data_time))
        if epoch % 50 == 0:
            torch.save(net.state_dict(), save_path+str(epoch)+'.pt')
            print('Model saved to %s' % (save_path+str(epoch)+'.pt'))

    print('fydnb!')