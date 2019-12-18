DATASET_URL = 'http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip'
DATA_PATH = './data/'
DATASET_FN = DATA_PATH + DATASET_URL.split('/')[-1]
DATASET_HDF = DATA_PATH + 'voxel_models.h5'

RANDOM_SEED = 42
CUBE_SIZE = 32
LEAK_VALUE = 0.2
Z_SIZE = 200
EPOCHS = 500
BATCH_SIZE = 360 # 360 for 6GB VRAM
DISCRIMINATOR_LEARNING_RATE = 0.001
GENERATOR_LEARNING_RATE = 0.0025
DISCRIMINATOR_THRESHOLD = 0.8
LOG_PATH = './log/'
GENERATED_PATH = './generated_models/'
CLASSES = ['wardrobe', 'bed', 'chair', 'laptop']
SAVED_DISCRIMINATOR = 'discriminator.pkl'
SAVED_GENERATOR = 'generator.pkl'

NUM_ROUTING_ITERATIONS = 3
NUM_CLASSES = len(CLASSES)
