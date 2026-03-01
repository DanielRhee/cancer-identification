import torch
import numpy as np

def getDevice():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def logitTransform(betaValues):
    return np.log(betaValues / (1 - betaValues))
