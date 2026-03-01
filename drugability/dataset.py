import torch
from torch.utils.data import Dataset

class TranslatorDataset(Dataset):
    def __init__(self, methylation, expression):
        self.methylation = methylation
        self.expression = expression

    def __len__(self):
        return len(self.methylation)

    def __getitem__(self, idx):
        return self.methylation[idx], self.expression[idx]

class PredictorDataset(Dataset):
    def __init__(self, expression, ic50, mask):
        self.expression = expression
        self.ic50 = ic50
        self.mask = mask

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        return self.expression[idx], self.ic50[idx], self.mask[idx]
