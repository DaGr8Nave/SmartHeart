import torch

DEVICE = "cuda"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_X = 'genh.pth.tar'
CHECKPOINT_GEN_Y = 'genz.pth.tar'
CHECKPOINT_CRITIC_X = 'critich.pth.tar'
CHECKPOINT_CRITIC_Y = 'criticz.pth.tar'