import torch

from gan import Generator
from constants import SAVED_GENERATOR

if __name__ == '__main__':
    # just save state_dict of the model
    model = torch.load(SAVED_GENERATOR, map_location='cpu')
    torch.save(model.state_dict(), SAVED_GENERATOR + '_state_dict')
