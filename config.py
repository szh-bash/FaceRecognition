# Configurations

# init
dataPath = {
'src': '/data/shenzhonghai/PIE_dataset/train',
'dst': '/data/shenzhonghai/PIE_dataset/train-112x112',
'Lfw': '/data/shenzhonghai/lfw/lfw',
'LfwDf': '/dev/shm/lfw-deepfunneled',
'MTLfw': '/data/shenzhonghai/lfw/mtcnn-lfw',
'ACLfw': '/dev/shm/Multi-Alined-Cropped-lfw',
'ACmtLfw': '/dev/shm/Multi-Alined-Cropped-mtcnn-lfw',
'ACLfwDf': '/dev/shm/Multi-Alined-Cropped-lfw-deepfunneled',
'ACLfwDf112P': '/dev/shm/Multi-Alined-Cropped-lfw-deepfunneled-112-perfect',
'MTLfwDf112P': '/data/shenzhonghai/mtcnn-lfw-112x112-perfect',
'MTLfwMix': '/data/shenzhonghai/mtcnn-lfw-mix',
'RetinaLfwWrap': '/data/shenzhonghai/retina-lfw-wrap',
'MegaLfw112': '/data/shenzhonghai/lfw-112x112',
'Web': '/data/shenzhonghai/CASIA-WebFace',
'MTWeb': '/dev/shm/mtcnn-CASIA-WebFace',
'ACWeb': '/dev/shm/Multi-Alined-Cropped-CASIA-WebFace',
'ACmtWeb': '/dev/shm/Multi-Alined-Cropped-mtcnn-CASIA-WebFace',
'ACWeb112': '/dev/shm/Multi-Alined-Cropped-CASIA-WebFace-112',
'MulACWeb112P': '/dev/shm/Multi-Alined-Cropped-CASIA-WebFace-112-Perfect',
'MTWeb112P': '/data/shenzhonghai/mtcnn-casia-112x112-perfect',
'MegaWeb112': '/data/shenzhonghai/casia-112x112',
'RetinaWebWrap': '/data/shenzhonghai/retina-casia-wrap',
'grimace': '/data/shenzhonghai/Face_Recognition_Data/grimace-112x112',
'faces96': '/data/shenzhonghai/Face_Recognition_Data/faces96-112x112',
'faces95': '/data/shenzhonghai/Face_Recognition_Data/faces95-112x112',
'faces94': '/data/shenzhonghai/Face_Recognition_Data/faces94-112x112',
'pie': '/data/shenzhonghai/PIE_dataset/test'
}

# train/test
Total = 72
batch_size = 512
test_batch_size = 1
learning_rate = 0.001
weight_decay = 0.0005
milestones = [36000, 54000]
# modelName = 'resnetFace50_lr3654_m40_112'
modelName = 'resnet_face50_MegaWeb112F_base5'
modelSavePath = '/data/shenzhonghai/FaceClustering/models/'+modelName
modelPath = '/data/shenzhonghai/FaceClustering/models/'+modelName+'.tar'
dp = 0.00
pairsTxtPath = '/data/shenzhonghai/lfw/pairs.txt'
# verificationPath = '/data/shenzhonghai/Face_Recognition_Data/grimace-pairs.txt'
# verificationPath = '/data/shenzhonghai/Face_Recognition_Data/faces96-pairs.txt'
# verificationPath = '/data/shenzhonghai/Face_Recognition_Data/faces95-pairs.txt'
verificationPath = '/data/shenzhonghai/PIE_dataset/pairs.txt'
server = 2332

# GCN
K = 2
