
# init
lfwPath = '/dev/shm/lfw'
lfwDfPath = '/dev/shm/lfw-deepfunneled'
webPath = '/dev/shm/CASIA-WebFace'

# train
Total = 35
batch_size = 128
learning_rate = 0.001
weight_decay = 0.001
modelSavePath = '/data/shenzhonghai/FaceClustering/models/Vgg16_base_aug_WD3_2|90k_ep'
# test
# modelPath = '/data/shenzhonghai/FaceClustering/models/Vgg16_wf_af-1_256_lr333_2|60k_ep30.pt'
modelPath = '/data/shenzhonghai/FaceClustering/models/Vgg16_base_aug_WD3_2|90k_ep35.pt'
# featPath = '/data/shenzhonghai/lfw/lfwdf-wf-af05-lr1e3-feat-fc2-ep15/'
featPath = '/data/shenzhonghai/lfw/lfwDf-base_aug_WD3-feat-fc2-ep35/'
pairsTxtPath = '/data/shenzhonghai/lfw/pairs.txt'
