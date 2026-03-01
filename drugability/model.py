import torch
import torch.nn as nn

class ExpressionTranslator(nn.Module):
    def __init__(self, numGenes, encoderDims=[5000, 2048, 1024, 256], decoderDims=[256, 1024, 2048], dropout=0.3):
        super().__init__()

        self.numGenes = numGenes

        encoderLayers = []
        inDim = encoderDims[0]
        for i, outDim in enumerate(encoderDims[1:]):
            encoderLayers.append(nn.Linear(inDim, outDim))
            if i < len(encoderDims) - 2:
                encoderLayers.append(nn.BatchNorm1d(outDim))
                encoderLayers.append(nn.GELU())
                encoderLayers.append(nn.Dropout(dropout))
            inDim = outDim

        self.encoder = nn.Sequential(*encoderLayers)

        decoderLayers = []
        inDim = decoderDims[0]
        for i, outDim in enumerate(decoderDims[1:]):
            decoderLayers.append(nn.Linear(inDim, outDim))
            decoderLayers.append(nn.BatchNorm1d(outDim))
            decoderLayers.append(nn.GELU())
            decoderLayers.append(nn.Dropout(dropout))
            inDim = outDim

        decoderLayers.append(nn.Linear(inDim, numGenes))

        self.decoder = nn.Sequential(*decoderLayers)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return reconstructed

class DrugPredictor(nn.Module):
    def __init__(self, numGenes, numDrugs, hiddenDims=[1024, 512, 256], dropout=0.4):
        super().__init__()

        layers = []
        inDim = numGenes

        for outDim in hiddenDims:
            layers.append(nn.Linear(inDim, outDim))
            layers.append(nn.BatchNorm1d(outDim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            inDim = outDim

        layers.append(nn.Linear(inDim, numDrugs))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
