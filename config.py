# Configurations

# init
lfwPath = '/dev/shm/lfw'
lfwDfPath = '/dev/shm/lfw-deepfunneled'
mtLfwPath = '/dev/shm/mtcnn-lfw'
webPath = '/dev/shm/CASIA-WebFace'
mtWebPath = '/dev/shm/mtcnn-CASIA-WebFace'

# train
Total = 80
batch_size = 128
learning_rate = 0.001
weight_decay = 0.00000
modelSavePath = '/data/shenzhonghai/FaceClustering/models/train_log_Vgg16_+256D_af-5e-2.pt'
# test
# modelPath = '/data/shenzhonghai/FaceClustering/models/Vgg16_wf_af-1_256_lr333_2|60k_ep30.pt'
# modelPath = '/data/shenzhonghai/FaceClustering/models/nVgg16_base_aug_DP05_2|90k_ep22.pt'
# modelPath = '/data/shenzhonghai/FaceClustering/models/Vgg16-T_base_2|200k_180000.pt'
# modelPath = '/data/shenzhonghai/FaceClustering/models/Vgg16_224_mtwfc_base_flip_2|200k_100000.pt'
modelPath = '/data/shenzhonghai/FaceClustering/models/train_log_Vgg16_+256D_af-5e-2.pt_150000.pt'
# featPath = '/data/shenzhonghai/lfw/lfwdf-wf-af05-lr1e3-feat-fc2-ep15/'
# featPath = '/data/shenzhonghai/lfw/lfwDf-base_aug_DP05-feat-fc2-ep32/'
# featPath = '/data/shenzhonghai/lfw/mtLfw-mtwfc-base-feat-fc2-35000/'
dp = 0.00
pairsTxtPath = '/data/shenzhonghai/lfw/pairs.txt'

# GCN
K = 2
