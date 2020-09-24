# Configurations

# init
lfwPath = '/dev/shm/lfw'
lfwDfPath = '/dev/shm/lfw-deepfunneled'
mtLfwPath = '/dev/shm/mtcnn-lfw'
webPath = '/dev/shm/CASIA-WebFace'
mtWebPath = '/dev/shm/mtcnn-CASIA-WebFace'

# train
Total = 200
batch_size = 256
learning_rate = 0.001
weight_decay = 0.00000
modelSavePath = '/data/shenzhonghai/FaceClustering/models/resnet50'
# test
modelPath = '/data/shenzhonghai/FaceClustering/models/resnet50.tar'
# featPath = '/data/shenzhonghai/lfw/lfwdf-wf-af05-lr1e3-feat-fc2-ep15/'
# featPath = '/data/shenzhonghai/lfw/lfwDf-base_aug_DP05-feat-fc2-ep32/'
# featPath = '/data/shenzhonghai/lfw/mtLfw-mtwfc-base-feat-fc2-35000/'
dp = 0.00
pairsTxtPath = '/data/shenzhonghai/lfw/pairs.txt'

# GCN
K = 2
