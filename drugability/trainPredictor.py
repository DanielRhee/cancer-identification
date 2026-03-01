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
from model import DrugPredictor
from dataset import PredictorDataset

def createDataLoaders():
    print("Loading preprocessed data...")

    trainExpr = torch.load(config.getArtifactPath('predictor_train_expression'), weights_only=False)
    valExpr = torch.load(config.getArtifactPath('predictor_val_expression'), weights_only=False)
    testExpr = torch.load(config.getArtifactPath('predictor_test_expression'), weights_only=False)

    trainIc50 = torch.load(config.getArtifactPath('predictor_train_ic50'), weights_only=False)
    valIc50 = torch.load(config.getArtifactPath('predictor_val_ic50'), weights_only=False)
    testIc50 = torch.load(config.getArtifactPath('predictor_test_ic50'), weights_only=False)

    trainMask = torch.load(config.getArtifactPath('predictor_train_mask'), weights_only=False)
    valMask = torch.load(config.getArtifactPath('predictor_val_mask'), weights_only=False)
    testMask = torch.load(config.getArtifactPath('predictor_test_mask'), weights_only=False)

    trainDataset = PredictorDataset(trainExpr, trainIc50, trainMask)
    valDataset = PredictorDataset(valExpr, valIc50, valMask)
    testDataset = PredictorDataset(testExpr, testIc50, testMask)

    trainLoader = DataLoader(trainDataset, batch_size=config.PREDICTOR_BATCH_SIZE, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=config.PREDICTOR_BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=config.PREDICTOR_BATCH_SIZE, shuffle=False)

    return trainLoader, valLoader, testLoader

def getLinearWarmupCosineSchedule(optimizer, warmupSteps, totalSteps):
    def lrLambda(currentStep):
        if currentStep < warmupSteps:
            return float(currentStep) / float(max(1, warmupSteps))
        progress = float(currentStep - warmupSteps) / float(max(1, totalSteps - warmupSteps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda)

def maskedMSELoss(pred, target, mask):
    squaredError = (pred - target) ** 2
    maskedError = squaredError * mask
    return maskedError.sum() / mask.sum()

def computePearsonR(pred, target, mask):
    maskedPred = pred * mask
    maskedTarget = target * mask

    predMean = maskedPred.sum(dim=0) / mask.sum(dim=0).clamp(min=1)
    targetMean = maskedTarget.sum(dim=0) / mask.sum(dim=0).clamp(min=1)

    predCentered = (maskedPred - predMean) * mask
    targetCentered = (maskedTarget - targetMean) * mask

    numerator = (predCentered * targetCentered).sum(dim=0)
    denominator = torch.sqrt((predCentered ** 2).sum(dim=0) * (targetCentered ** 2).sum(dim=0))

    r = numerator / (denominator + 1e-8)

    validDrugs = mask.sum(dim=0) > 1
    r = torch.where(validDrugs, r, torch.zeros_like(r))

    return r

def trainEpoch(model, dataLoader, optimizer, scheduler, device):
    model.train()
    totalLoss = 0

    for expression, ic50, mask in dataLoader:
        expression, ic50, mask = expression.to(device), ic50.to(device), mask.to(device)

        optimizer.zero_grad()
        predIc50 = model(expression)
        loss = maskedMSELoss(predIc50, ic50, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        totalLoss += loss.item()

    avgLoss = totalLoss / len(dataLoader)
    return avgLoss

def evaluateModel(model, dataLoader, device):
    model.eval()
    totalLoss = 0
    allPreds = []
    allTargets = []
    allMasks = []

    with torch.no_grad():
        for expression, ic50, mask in dataLoader:
            expression, ic50, mask = expression.to(device), ic50.to(device), mask.to(device)

            predIc50 = model(expression)
            loss = maskedMSELoss(predIc50, ic50, mask)

            totalLoss += loss.item()
            allPreds.append(predIc50.cpu())
            allTargets.append(ic50.cpu())
            allMasks.append(mask.cpu())

    avgLoss = totalLoss / len(dataLoader)

    allPreds = torch.cat(allPreds, dim=0)
    allTargets = torch.cat(allTargets, dim=0)
    allMasks = torch.cat(allMasks, dim=0)

    perDrugR = computePearsonR(allPreds, allTargets, allMasks)

    validR = perDrugR[perDrugR != 0]
    if len(validR) > 0:
        meanR = validR.mean().item()
        medianR = validR.median().item()
        fracAbove03 = (validR > 0.3).float().mean().item()
    else:
        meanR = medianR = fracAbove03 = 0.0

    return avgLoss, meanR, medianR, fracAbove03, perDrugR

def train():
    print("=" * 60)
    print("Model 3: Drug Predictor Training")
    print("=" * 60)

    device = getDevice()
    print(f"Using device: {device}")

    trainLoader, valLoader, testLoader = createDataLoaders()

    with open(config.getArtifactPath('gene_overlap'), 'r') as f:
        geneOverlap = json.load(f)
    numGenes = len(geneOverlap)

    with open(config.getArtifactPath('drug_names'), 'r') as f:
        drugNames = json.load(f)
    numDrugs = len(drugNames)

    print(f"Building model with {numGenes} genes and {numDrugs} drugs...")
    model = DrugPredictor(
        numGenes=numGenes,
        numDrugs=numDrugs,
        hiddenDims=config.PREDICTOR_HIDDEN_DIMS,
        dropout=config.PREDICTOR_DROPOUT
    ).to(device)

    totalParams = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {totalParams:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.PREDICTOR_LEARNING_RATE,
        weight_decay=config.PREDICTOR_WEIGHT_DECAY
    )

    warmupSteps = config.PREDICTOR_WARMUP_EPOCHS * len(trainLoader)
    totalSteps = config.PREDICTOR_NUM_EPOCHS * len(trainLoader)
    scheduler = getLinearWarmupCosineSchedule(optimizer, warmupSteps, totalSteps)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {config.PREDICTOR_NUM_EPOCHS}")
    print(f"  Batch size: {config.PREDICTOR_BATCH_SIZE}")
    print(f"  Learning rate: {config.PREDICTOR_LEARNING_RATE}")
    print(f"  Weight decay: {config.PREDICTOR_WEIGHT_DECAY}")
    print(f"  Warmup epochs: {config.PREDICTOR_WARMUP_EPOCHS}")
    print(f"  Early stopping patience: {config.PREDICTOR_PATIENCE}")
    print()

    bestMeanR = -float('inf')
    patience = config.PREDICTOR_PATIENCE
    patienceCounter = 0

    for epoch in range(1, config.PREDICTOR_NUM_EPOCHS + 1):
        trainLoss = trainEpoch(model, trainLoader, optimizer, scheduler, device)

        valLoss, valMeanR, valMedianR, valFracAbove03, _ = evaluateModel(model, valLoader, device)

        print(f"Epoch {epoch:3d}/{config.PREDICTOR_NUM_EPOCHS} | "
              f"Train Loss: {trainLoss:.4f} | "
              f"Val Loss: {valLoss:.4f} | "
              f"Val Mean R: {valMeanR:.4f} | "
              f"Val Median R: {valMedianR:.4f} | "
              f"Val >0.3: {valFracAbove03:.3f}")

        if valMeanR > bestMeanR:
            bestMeanR = valMeanR
            patienceCounter = 0
            torch.save(model.state_dict(), config.getArtifactPath('predictor_best_model'))
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
        model.load_state_dict(torch.load(config.getArtifactPath('predictor_best_model'), weights_only=True))
    else:
        print("No improvement during training, using final model state...")

    testLoss, testMeanR, testMedianR, testFracAbove03, testPerDrugR = evaluateModel(model, testLoader, device)

    print(f"\nTest Set Performance:")
    print(f"  Loss: {testLoss:.4f}")
    print(f"  Mean Pearson R: {testMeanR:.4f}")
    print(f"  Median Pearson R: {testMedianR:.4f}")
    print(f"  Fraction >0.3: {testFracAbove03:.3f}")

    validR = testPerDrugR[testPerDrugR != 0]
    print(f"\nPer-drug Pearson R distribution ({len(validR)} drugs):")
    print(f"  Min: {validR.min():.4f}")
    print(f"  25th percentile: {validR.quantile(0.25):.4f}")
    print(f"  50th percentile: {validR.quantile(0.50):.4f}")
    print(f"  75th percentile: {validR.quantile(0.75):.4f}")
    print(f"  Max: {validR.max():.4f}")

    perDrugReliability = {drugNames[i]: float(testPerDrugR[i]) for i in range(len(drugNames))}
    with open(config.ARTIFACTS_DIR / 'perDrugReliability.json', 'w') as f:
        json.dump(perDrugReliability, f, indent=2)

    print("\nPer-drug reliability scores saved to artifacts/perDrugReliability.json")

if __name__ == '__main__':
    train()
