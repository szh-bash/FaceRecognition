# Configurations

# init
lfwPath = '/dev/shm/lfw'
lfwDfPath = '/dev/shm/lfw-deepfunneled'
mtLfwPath = '/dev/shm/mtcnn-lfw'
ACLfwPath = '/dev/shm/Alined-Cropped-lfw'
MulACLfwPath = '/dev/shm/Multi-Alined-Cropped-lfw'
MulACmtLfwPath = '/dev/shm/Multi-Alined-Cropped-mtcnn-lfw'
MulACLfwDfPath = '/dev/shm/Multi-Alined-Cropped-lfw-deepfunneled'
webPath = '/dev/shm/CASIA-WebFace'
mtWebPath = '/dev/shm/mtcnn-CASIA-WebFace'
ACWebPath = '/dev/shm/Alined-Cropped-CASIA-WebFace'
MulACWebPath = '/dev/shm/Multi-Alined-Cropped-CASIA-WebFace'
MulACmtWebPath = '/dev/shm/Multi-Alined-Cropped-mtcnn-CASIA-WebFace'

# train
Total = 30
batch_size = 256
learning_rate = 0.001
weight_decay = 0.00000
modelName = 'resnet50_ACmtWebFace_base*'
modelSavePath = '/data/shenzhonghai/FaceClustering/models/'+modelName
# test
modelPath = '/data/shenzhonghai/FaceClustering/models/'+modelName+'.tar'
# featPath = '/data/shenzhonghai/lfw/lfwdf-wf-af05-lr1e3-feat-fc2-ep15/'
# featPath = '/data/shenzhonghai/lfw/lfwDf-base_aug_DP05-feat-fc2-ep32/'
# featPath = '/data/shenzhonghai/lfw/mtLfw-mtwfc-base-feat-fc2-35000/'
dp = 0.00
pairsTxtPath = '/data/shenzhonghai/lfw/pairs.txt'
server = 2333

# GCN
K = 2

