import numpy as np
# Configurations

dataPath = {
'src': '/data/shenzhonghai/PIE_dataset/pie29',
'dst': '/data/shenzhonghai/PIE_dataset/pie29-112x112',
'Lfw': '/data/shenzhonghai/lfw/lfw',
'MegaLfw112': '/data/shenzhonghai/lfw-112x112',
'RetinaLfwCenter': '/data/shenzhonghai/retina-lfw-center',
'Web': '/data/shenzhonghai/CASIA-WebFace',
'MegaWeb112': '/data/shenzhonghai/casia-112x112',
'RetinaWebCenter': '/data/shenzhonghai/retina-casia-center',
'pie': '/data/shenzhonghai/PIE_dataset/pie-112x112',
'pieC': '/data/shenzhonghai/PIE_dataset/test-112x112-center',
'pieCC': '/data/shenzhonghai/PIE_dataset/test-112x112-center-clean',
'faces94C': '/data/shenzhonghai/Face_Recognition_Data/faces94-112x112-center',
'faces95C': '/data/shenzhonghai/Face_Recognition_Data/faces95-112x112-center',
'faces96C': '/data/shenzhonghai/Face_Recognition_Data/faces96-112x112-center',
'grimaceC': '/data/shenzhonghai/Face_Recognition_Data/grimace-112x112-center'
}

batch_size = 512
# Total = 36 * batch_size // 256
Total = 36
test_batch_size = 32
momentum = 0.9
# learning_rate = 0.001
learning_rate = 0.1
weight_decay = 0.0005
# milestones = np.array([36000, 54000])
milestones = np.array([20000, 28000])
# modelName = 'resnet_face50_RetinaWeb112FC_m45'
# modelName = 'resnet_face50_MegaWeb112F_base5'
# modelName = 'resnet_face50_RetinaWeb112FC_m45_border0_cutout'
# modelName = 'resnet_face50_RetinaWeb112FC_m50_border0'
# modelName = 'resnet_face50_RetinaWeb112FC_m45_border0_cutout'
modelName = 'sgd_s32_m50_flip_gb_cutout_trs_rot'
modelSavePath = '/data/shenzhonghai/FaceClustering/models/'+modelName
modelPath = '/data/shenzhonghai/FaceClustering/models/'+modelName+'.tar'
dp = 0.00

server = 2333
verificationPath = {
    'MegaLfw112': '/data/shenzhonghai/lfw/pairs.txt',
    'RetinaLfwWrap': '/data/shenzhonghai/lfw/pairs.txt',
    'RetinaLfwCenter': '/data/shenzhonghai/lfw/pairs.txt',
    'pieC': '/data/shenzhonghai/PIE_dataset/pie-pairs.txt',
    'pieCC': '/data/shenzhonghai/PIE_dataset/pie-clean-pairs.txt',
    'faces94C': '/data/shenzhonghai/Face_Recognition_Data/faces94-pairs.txt',
    'faces95C': '/data/shenzhonghai/Face_Recognition_Data/faces95-pairs.txt',
    'faces96C': '/data/shenzhonghai/Face_Recognition_Data/faces96-pairs.txt',
    'grimaceC': '/data/shenzhonghai/Face_Recognition_Data/grimace-pairs.txt',
}
