import torch
from torch.utils.serialization import load_lua


import sys
sys.path.append('..')
from config import vgg16_ox

x = load_lua(vgg16_ox, unknown_classes=True)
# pickle.load = partial(pickle.load())
# x = Vgg16('arcFace', 'lfwDf')
# checkpoint = torch.load(vgg16_ox)
# x.load_state_dict(checkpoint['net_dict'])
print(x)
