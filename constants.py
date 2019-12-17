DATASET_URL = 'http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip'
DATA_PATH = './data/'
DATASET_FN = DATA_PATH + DATASET_URL.split('/')[-1]
DATASET_HDF = DATA_PATH + 'voxel_models.h5'

LATENT_SIZE = 200
CUBE_SIZE = 32
LEAK_VALUE = 0.2
Z_SIZE = 200
EPOCHS = 500
BATCH_SIZE = 8 # 350 for 6GB VRAM
D_LR = 0.001
G_LR = 0.0025
D_THRESH = 0.8
LOG_PATH = './log/'
GENERATED_PATH = './generated_models/'
CATEGORIES = ['wardrobe', 'xbox']
