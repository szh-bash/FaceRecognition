import numpy as np
# Configurations

# init
dataPath = {
'src': '/data/shenzhonghai/PIE_dataset/test',
'dst': '/data/shenzhonghai/PIE_dataset/test-112x112-center',
'Lfw': '/data/shenzhonghai/lfw/lfw',
'MegaLfw112': '/data/shenzhonghai/lfw-112x112',
'RetinaLfwCenter': '/data/shenzhonghai/retina-lfw-center',
'Web': '/data/shenzhonghai/CASIA-WebFace',
'MegaWeb112': '/data/shenzhonghai/casia-112x112',
'RetinaWebCenter': '/data/shenzhonghai/retina-casia-center',
'faces94': '/data/shenzhonghai/Face_Recognition_Data/faces94-112x112',
'pieC': '/data/shenzhonghai/PIE_dataset/test-112x112-center',
'faces94C': '/data/shenzhonghai/Face_Recognition_Data/faces94-112x112-center',
'faces95C': '/data/shenzhonghai/Face_Recognition_Data/faces95-112x112-center',
'faces96C': '/data/shenzhonghai/Face_Recognition_Data/faces96-112x112-center',
'grimaceC': '/data/shenzhonghai/Face_Recognition_Data/grimace-112x112-center'
}

# train/test
Total = 72
batch_size = 512
test_batch_size = 32
learning_rate = 0.001
weight_decay = 0.0005
milestones = np.array([36000, 54000]) * 512 // batch_size
# modelName = 'resnetFace50_lr3654_m40_112'
# modelName = 'resnet_face50_RetinaWeb112FC_m45'
# modelName = 'resnet_face50_RetinaWeb112FC_base5'
# modelName = 'resnet_face50_RetinaWeb112F_base5'
modelName = 'resnet_face50_MegaWeb112F_base5'
modelSavePath = '/data/shenzhonghai/FaceClustering/models/'+modelName
modelPath = '/data/shenzhonghai/FaceClustering/models/'+modelName+'.tar'
dp = 0.00
verificationPath = {
    'MegaLfw112': '/data/shenzhonghai/lfw/pairs.txt',
    'RetinaLfwWrap': '/data/shenzhonghai/lfw/pairs.txt',
    'RetinaLfwCenter': '/data/shenzhonghai/lfw/pairs.txt',
    'grimaceC': '/data/shenzhonghai/Face_Recognition_Data/grimace-pairs.txt',
    'faces96C': '/data/shenzhonghai/Face_Recognition_Data/faces96-pairs.txt',
    'faces95C': '/data/shenzhonghai/Face_Recognition_Data/faces95-pairs.txt',
    'pieC': '/data/shenzhonghai/PIE_dataset/pie-pairs.txt'
}
server = 2332

# GCN
K = 2
