
# init
lfwPath = '/dev/shm/lfw'
webPath = '/dev/shm/CASIA-WebFace'

# train
Total = 35
batch_size = 128
learning_rate = 0.001
modelSavePath = '/data/shenzhonghai/FaceClustering/models/Vgg16_wfP_af05_64_128_lr1e3_2|60k_ep'

# test
featPath = '/data/shenzhonghai/lfw/wf-af05-lr2e3-feat-fc2-ep20/'
pairsTxtPath = '/data/shenzhonghai/lfw/pairs.txt'
modelPath = '/data/shenzhonghai/FaceClustering/models/Vgg16_wfP_af05_64_128_lr1e3_2|60k_ep20.pt'