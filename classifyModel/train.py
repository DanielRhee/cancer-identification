import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import json

import config
from utils import getDevice
from dataset import CancerDataset
from model import CancerClassifier

def createDataLoaders():
    print("Loading preprocessed data...")

    trainData = torch.load(config.getArtifactPath('train_data'), weights_only=False)
    valData = torch.load(config.getArtifactPath('val_data'), weights_only=False)
    testData = torch.load(config.getArtifactPath('test_data'), weights_only=False)

    trainLabels = torch.load(config.getArtifactPath('train_labels'), weights_only=False)
    valLabels = torch.load(config.getArtifactPath('val_labels'), weights_only=False)
    testLabels = torch.load(config.getArtifactPath('test_labels'), weights_only=False)

    trainDataset = CancerDataset(trainData, trainLabels)
    valDataset = CancerDataset(valData, valLabels)
    testDataset = CancerDataset(testData, testLabels)

    trainLoader = DataLoader(trainDataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=config.BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return trainLoader, valLoader, testLoader

def getLinearWarmupCosineSchedule(optimizer, warmupSteps, totalSteps):
    def lrLambda(currentStep):
        if currentStep < warmupSteps:
            return float(currentStep) / float(max(1, warmupSteps))
        progress = float(currentStep - warmupSteps) / float(max(1, totalSteps - warmupSteps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda)

def trainEpoch(model, dataLoader, criterion, optimizer, scheduler, device):
    model.train()
    totalLoss = 0
    allPreds = []
    allLabels = []

    for features, labels in dataLoader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        totalLoss += loss.item()
        preds = outputs.argmax(dim=1)
        allPreds.extend(preds.cpu().numpy())
        allLabels.extend(labels.cpu().numpy())

    avgLoss = totalLoss / len(dataLoader)
    accuracy = accuracy_score(allLabels, allPreds)
    macroF1 = f1_score(allLabels, allPreds, average='macro')

    return avgLoss, accuracy, macroF1

def evaluateModel(model, dataLoader, criterion, device):
    model.eval()
    totalLoss = 0
    allPreds = []
    allLabels = []

    with torch.no_grad():
        for features, labels in dataLoader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            totalLoss += loss.item()
            preds = outputs.argmax(dim=1)
            allPreds.extend(preds.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())

    avgLoss = totalLoss / len(dataLoader)
    accuracy = accuracy_score(allLabels, allPreds)
    macroF1 = f1_score(allLabels, allPreds, average='macro')

    return avgLoss, accuracy, macroF1, allPreds, allLabels

def plotConfusionMatrix(labels, preds, labelEncoder):
    inverseEncoder = {v: k for k, v in labelEncoder.items()}
    classNames = [inverseEncoder[i] for i in range(len(labelEncoder))]

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classNames, yticklabels=classNames)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(config.getArtifactPath('confusion_matrix'), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {config.getArtifactPath('confusion_matrix')}")

def train():
    print("=" * 60)
    print("Cancer Type Classifier - Training")
    print("=" * 60)

    device = getDevice()
    print(f"Using device: {device}")

    trainLoader, valLoader, testLoader = createDataLoaders()

    with open(config.getArtifactPath('label_encoder'), 'r') as f:
        labelEncoder = json.load(f)

    classWeights = torch.load(config.getArtifactPath('class_weights'), weights_only=False).to(device)

    inputDim = trainLoader.dataset.features.shape[1]
    numClasses = len(labelEncoder)

    print(f"\nModel configuration:")
    print(f"  Input dimension: {inputDim}")
    print(f"  Hidden dimensions: {config.HIDDEN_DIMS}")
    print(f"  Number of classes: {numClasses}")
    print(f"  Dropout rate: {config.DROPOUT_RATE}")

    model = CancerClassifier(inputDim, numClasses).to(device)

    totalParams = sum(p.numel() for p in model.parameters())
    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {totalParams:,}")
    print(f"  Trainable parameters: {trainableParams:,}")

    criterion = nn.CrossEntropyLoss(weight=classWeights, label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    stepsPerEpoch = len(trainLoader)
    warmupSteps = config.WARMUP_EPOCHS * stepsPerEpoch
    totalSteps = config.NUM_EPOCHS * stepsPerEpoch
    scheduler = getLinearWarmupCosineSchedule(optimizer, warmupSteps, totalSteps)

    bestValF1 = 0
    bestEpoch = 0
    patience = 0

    print(f"\nStarting training for up to {config.NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 60)

        trainLoss, trainAcc, trainF1 = trainEpoch(model, trainLoader, criterion, optimizer, scheduler, device)

        valLoss, valAcc, valF1, _, _ = evaluateModel(model, valLoader, criterion, device)

        currentLr = optimizer.param_groups[0]['lr']

        print(f"  Train - Loss: {trainLoss:.4f} | Acc: {trainAcc:.4f} | Macro-F1: {trainF1:.4f}")
        print(f"  Val   - Loss: {valLoss:.4f} | Acc: {valAcc:.4f} | Macro-F1: {valF1:.4f}")
        print(f"  Learning Rate: {currentLr:.6f}")

        if valF1 > bestValF1:
            bestValF1 = valF1
            bestEpoch = epoch + 1
            patience = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': valF1,
                'val_loss': valLoss,
                'val_acc': valAcc
            }, config.getArtifactPath('best_model'))

            print(f"  *** New best model saved (Val Macro-F1: {valF1:.4f}) ***")
        else:
            patience += 1
            print(f"  Patience: {patience}/{config.EARLY_STOPPING_PATIENCE}")

            if patience >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Val Macro-F1: {bestValF1:.4f} at epoch {bestEpoch}")
    print("=" * 60)

    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(config.getArtifactPath('best_model'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    testLoss, testAcc, testF1, testPreds, testLabels = evaluateModel(model, testLoader, criterion, device)

    print(f"\nTest Set Results:")
    print(f"  Loss: {testLoss:.4f}")
    print(f"  Accuracy: {testAcc:.4f}")
    print(f"  Macro-F1: {testF1:.4f}")

    inverseEncoder = {v: k for k, v in labelEncoder.items()}
    classNames = [inverseEncoder[i] for i in range(len(labelEncoder))]

    print("\n" + "=" * 60)
    print("Classification Report (Test Set)")
    print("=" * 60)
    print(classification_report(testLabels, testPreds, target_names=classNames, digits=4))

    plotConfusionMatrix(testLabels, testPreds, labelEncoder)

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print(f"Best model saved to: {config.getArtifactPath('best_model')}")
    print(f"Confusion matrix saved to: {config.getArtifactPath('confusion_matrix')}")
    print("=" * 60)

if __name__ == '__main__':
    train()
