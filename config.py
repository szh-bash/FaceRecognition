# Configurations

# init
lfwPath = '/dev/shm/lfw'
lfwDfPath = '/dev/shm/lfw-deepfunneled'
mtLfwPath = '/dev/shm/mtcnn-lfw'
webPath = '/dev/shm/CASIA-WebFace'
mtWebPath = '/dev/shm/mtcnn-CASIA-WebFace'

# ox-model
vgg16_ox = '/home/shenzhonghai/vgg_face_torch/VGG_FACE.t7'

# train
Total = 40
batch_size = 128
learning_rate = 0.001
weight_decay = 0.00000
modelSavePath = '/data/shenzhonghai/FaceClustering/models/Vgg16_mtwf_base_2|100k_ep'
# test
# modelPath = '/data/shenzhonghai/FaceClustering/models/Vgg16_wf_af-1_256_lr333_2|60k_ep30.pt'
# modelPath = '/data/shenzhonghai/FaceClustering/models/nVgg16_base_aug_DP05_2|90k_ep22.pt'
modelPath = '/data/shenzhonghai/FaceClustering/models/Vgg16_mtwf_base_2|100k_ep0.tar'
# featPath = '/data/shenzhonghai/lfw/lfwdf-wf-af05-lr1e3-feat-fc2-ep15/'
# featPath = '/data/shenzhonghai/lfw/lfwDf-base_aug_DP05-feat-fc2-ep32/'
featPath = '/data/shenzhonghai/lfw/mtlfw-mtwf-base-feat-fc2-ep0/'
dp = 0.00
pairsTxtPath = '/data/shenzhonghai/lfw/pairs.txt'

# GCN
K = 2
