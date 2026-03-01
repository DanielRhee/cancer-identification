import torch
import torch.nn as nn
import config

class ResidualBlock(nn.Module):
    def __init__(self, inDim, outDim, dropout=config.DROPOUT_RATE):
        super().__init__()
        self.linear = nn.Linear(inDim, outDim)
        self.bn = nn.BatchNorm1d(outDim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class CancerClassifier(nn.Module):
    def __init__(self, inputDim, numClasses, hiddenDims=None):
        super().__init__()

        if hiddenDims is None:
            hiddenDims = config.HIDDEN_DIMS

        self.blocks = nn.ModuleList()
        self.skipProjections = nn.ModuleList()

        prevDim = inputDim
        for i, hiddenDim in enumerate(hiddenDims):
            self.blocks.append(ResidualBlock(prevDim, hiddenDim))

            if i % 2 == 1 and i > 0:
                inputDimForSkip = hiddenDims[i-2] if i >= 2 else inputDim
                self.skipProjections.append(nn.Linear(inputDimForSkip, hiddenDim))
            else:
                self.skipProjections.append(None)

            prevDim = hiddenDim

        self.head = nn.Linear(hiddenDims[-1], numClasses)

    def forward(self, x):
        skipInputs = []

        for i, (block, skipProj) in enumerate(zip(self.blocks, self.skipProjections)):
            if i % 2 == 0:
                skipInputs.append(x)

            x = block(x)

            if skipProj is not None:
                skipIdx = i // 2
                if skipIdx < len(skipInputs):
                    skip = skipProj(skipInputs[skipIdx])
                    x = x + skip

        x = self.head(x)
        return x
