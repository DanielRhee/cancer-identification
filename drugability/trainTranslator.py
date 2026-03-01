import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import sys
from pathlib import Path

parentDir = Path(__file__).parent.parent
sys.path.append(str(parentDir))
sys.path.append(str(parentDir / 'classifyModel'))

from classifyModel.utils import getDevice

import config
from model import ExpressionTranslator
from dataset import TranslatorDataset

def createDataLoaders():
    print("Loading preprocessed data...")

    trainMeth = torch.load(config.getArtifactPath('translator_train_methylation'), weights_only=False)
    valMeth = torch.load(config.getArtifactPath('translator_val_methylation'), weights_only=False)
    testMeth = torch.load(config.getArtifactPath('translator_test_methylation'), weights_only=False)

    trainExpr = torch.load(config.getArtifactPath('translator_train_expression'), weights_only=False)
    valExpr = torch.load(config.getArtifactPath('translator_val_expression'), weights_only=False)
    testExpr = torch.load(config.getArtifactPath('translator_test_expression'), weights_only=False)

    trainDataset = TranslatorDataset(trainMeth, trainExpr)
    valDataset = TranslatorDataset(valMeth, valExpr)
    testDataset = TranslatorDataset(testMeth, testExpr)

    trainLoader = DataLoader(trainDataset, batch_size=config.TRANSLATOR_BATCH_SIZE, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=config.TRANSLATOR_BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=config.TRANSLATOR_BATCH_SIZE, shuffle=False)

    return trainLoader, valLoader, testLoader

def getLinearWarmupCosineSchedule(optimizer, warmupSteps, totalSteps):
    def lrLambda(currentStep):
        if currentStep < warmupSteps:
            return float(currentStep) / float(max(1, warmupSteps))
        progress = float(currentStep - warmupSteps) / float(max(1, totalSteps - warmupSteps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda)

def computePearsonR(pred, target):
    predMean = pred.mean(dim=0)
    targetMean = target.mean(dim=0)

    predCentered = pred - predMean
    targetCentered = target - targetMean

    numerator = (predCentered * targetCentered).sum(dim=0)
    denominator = torch.sqrt((predCentered ** 2).sum(dim=0) * (targetCentered ** 2).sum(dim=0))

    r = numerator / (denominator + 1e-8)

    return r

def trainEpoch(model, dataLoader, criterion, optimizer, scheduler, device):
    model.train()
    totalLoss = 0

    for methylation, expression in dataLoader:
        methylation, expression = methylation.to(device), expression.to(device)

        optimizer.zero_grad()
        predExpression = model(methylation)
        loss = criterion(predExpression, expression)
        loss.backward()
        optimizer.step()
        scheduler.step()

        totalLoss += loss.item()

    avgLoss = totalLoss / len(dataLoader)
    return avgLoss

def evaluateModel(model, dataLoader, criterion, device):
    model.eval()
    totalLoss = 0
    allPreds = []
    allTargets = []

    with torch.no_grad():
        for methylation, expression in dataLoader:
            methylation, expression = methylation.to(device), expression.to(device)

            predExpression = model(methylation)
            loss = criterion(predExpression, expression)

            totalLoss += loss.item()
            allPreds.append(predExpression.cpu())
            allTargets.append(expression.cpu())

    avgLoss = totalLoss / len(dataLoader)

    allPreds = torch.cat(allPreds, dim=0)
    allTargets = torch.cat(allTargets, dim=0)

    perGeneR = computePearsonR(allPreds, allTargets)

    meanR = perGeneR.mean().item()
    medianR = perGeneR.median().item()
    fracAbove03 = (perGeneR > 0.3).float().mean().item()

    return avgLoss, meanR, medianR, fracAbove03, perGeneR

def train():
    print("=" * 60)
    print("Model 2: Expression Translator Training")
    print("=" * 60)

    device = getDevice()
    print(f"Using device: {device}")

    trainLoader, valLoader, testLoader = createDataLoaders()

    with open(config.getArtifactPath('gene_overlap'), 'r') as f:
        geneOverlap = json.load(f)
    numGenes = len(geneOverlap)

    print(f"Building model with {numGenes} genes...")
    model = ExpressionTranslator(
        numGenes=numGenes,
        encoderDims=config.TRANSLATOR_ENCODER_DIMS,
        decoderDims=config.TRANSLATOR_DECODER_DIMS,
        dropout=config.TRANSLATOR_DROPOUT
    ).to(device)

    totalParams = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {totalParams:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRANSLATOR_LEARNING_RATE,
        weight_decay=config.TRANSLATOR_WEIGHT_DECAY
    )

    warmupSteps = config.TRANSLATOR_WARMUP_EPOCHS * len(trainLoader)
    totalSteps = config.TRANSLATOR_NUM_EPOCHS * len(trainLoader)
    scheduler = getLinearWarmupCosineSchedule(optimizer, warmupSteps, totalSteps)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {config.TRANSLATOR_NUM_EPOCHS}")
    print(f"  Batch size: {config.TRANSLATOR_BATCH_SIZE}")
    print(f"  Learning rate: {config.TRANSLATOR_LEARNING_RATE}")
    print(f"  Weight decay: {config.TRANSLATOR_WEIGHT_DECAY}")
    print(f"  Warmup epochs: {config.TRANSLATOR_WARMUP_EPOCHS}")
    print(f"  Early stopping patience: {config.TRANSLATOR_PATIENCE}")
    print()

    bestMeanR = -float('inf')
    patience = config.TRANSLATOR_PATIENCE
    patienceCounter = 0

    for epoch in range(1, config.TRANSLATOR_NUM_EPOCHS + 1):
        trainLoss = trainEpoch(model, trainLoader, criterion, optimizer, scheduler, device)

        valLoss, valMeanR, valMedianR, valFracAbove03, _ = evaluateModel(model, valLoader, criterion, device)

        print(f"Epoch {epoch:3d}/{config.TRANSLATOR_NUM_EPOCHS} | "
              f"Train Loss: {trainLoss:.4f} | "
              f"Val Loss: {valLoss:.4f} | "
              f"Val Mean R: {valMeanR:.4f} | "
              f"Val Median R: {valMedianR:.4f} | "
              f"Val >0.3: {valFracAbove03:.3f}")

        if valMeanR > bestMeanR:
            bestMeanR = valMeanR
            patienceCounter = 0
            torch.save(model.state_dict(), config.getArtifactPath('translator_best_model'))
            print(f"  *** New best model saved (Mean R: {bestMeanR:.4f}) ***")
        else:
            patienceCounter += 1
            if patienceCounter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if bestMeanR > -float('inf'):
        print("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(config.getArtifactPath('translator_best_model'), weights_only=True))
    else:
        print("No improvement during training, using final model state...")

    testLoss, testMeanR, testMedianR, testFracAbove03, testPerGeneR = evaluateModel(model, testLoader, criterion, device)

    print(f"\nTest Set Performance:")
    print(f"  Loss: {testLoss:.4f}")
    print(f"  Mean Pearson R: {testMeanR:.4f}")
    print(f"  Median Pearson R: {testMedianR:.4f}")
    print(f"  Fraction >0.3: {testFracAbove03:.3f}")

    print(f"\nPer-gene Pearson R distribution:")
    print(f"  Min: {testPerGeneR.min():.4f}")
    print(f"  25th percentile: {testPerGeneR.quantile(0.25):.4f}")
    print(f"  50th percentile: {testPerGeneR.quantile(0.50):.4f}")
    print(f"  75th percentile: {testPerGeneR.quantile(0.75):.4f}")
    print(f"  Max: {testPerGeneR.max():.4f}")

if __name__ == '__main__':
    train()
