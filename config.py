# Configurations

# init
dataPath = {
'Lfw': '/dev/shm/lfw',
'LfwDf': '/dev/shm/lfw-deepfunneled',
'MTLfw': '/dev/shm/mtcnn-lfw',
'ACLfw': '/dev/shm/Alined-Cropped-lfw',
'MulACLfw': '/dev/shm/Multi-Alined-Cropped-lfw',
'MulACmtLfw': '/dev/shm/Multi-Alined-Cropped-mtcnn-lfw',
'MulACLfwDf': '/dev/shm/Multi-Alined-Cropped-lfw-deepfunneled',
'MegaLfw112': '/data/shenzhonghai/lfw-112x112',
'Web': '/dev/shm/CASIA-WebFace',
'MTWeb': '/dev/shm/mtcnn-CASIA-WebFace',
'ACWeb': '/dev/shm/Alined-Cropped-CASIA-WebFace',
'MulACWeb': '/dev/shm/Multi-Alined-Cropped-CASIA-WebFace',
'MulACmtWeb': '/dev/shm/Multi-Alined-Cropped-mtcnn-CASIA-WebFace',
'MulACWeb112': '/dev/shm/Multi-Alined-Cropped-CASIA-WebFace-112',
'MegaWeb112': '/data/shenzhonghai/casia-112x112'
}

# train/test
Total = 30
batch_size = 512
learning_rate = 0.001
weight_decay = 0.0005
modelName = 'resnet_face50_MegaWebFace112_base1'
modelSavePath = '/data/shenzhonghai/FaceClustering/models/'+modelName
modelPath = '/data/shenzhonghai/FaceClustering/models/'+modelName+'.tar'
dp = 0.00
pairsTxtPath = '/data/shenzhonghai/lfw/pairs.txt'
server = 2333

# GCN
K = 2

