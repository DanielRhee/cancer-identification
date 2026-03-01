import argparse
import json
from pathlib import Path

import torch
import pandas as pd
import numpy as np

import config
from utils import getDevice, logitTransform
from model import CancerClassifier

def loadArtifacts():
    print("Loading trained model and preprocessing parameters...")

    with open(config.getArtifactPath('label_encoder'), 'r', encoding = 'utf-8') as f:
        labelEncoder = json.load(f)

    with open(config.getArtifactPath('probe_names'), 'r', encoding = 'utf-8') as f:
        probeNames = json.load(f)

    imputeMedians = torch.load(config.getArtifactPath('impute_medians')).numpy()
    featureMean = torch.load(config.getArtifactPath('feature_mean')).numpy()
    featureStd = torch.load(config.getArtifactPath('feature_std')).numpy()

    inverseEncoder = {v: k for k, v in labelEncoder.items()}

    inputDim = len(probeNames)
    numClasses = len(labelEncoder)

    device = getDevice()
    model = CancerClassifier(inputDim, numClasses).to(device)

    checkpoint = torch.load(config.getArtifactPath('best_model'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Validation Macro-F1: {checkpoint['val_f1']:.4f}")
    print(f"  Using device: {device}")

    return model, probeNames, imputeMedians, featureMean, featureStd, inverseEncoder, device

def preprocessSample(sampleData, probeNames, imputeMedians, featureMean, featureStd):
    if isinstance(sampleData, str):
        samplePath = Path(sampleData)
        if samplePath.suffix == '.tsv':
            sample = pd.read_csv(samplePath, sep='\t', index_col=0).iloc[0]
        else:
            sample = pd.read_csv(samplePath, index_col=0).iloc[0]
    else:
        sample = sampleData

    features = np.zeros(len(probeNames), dtype=np.float32)

    for i, probeName in enumerate(probeNames):
        if probeName in sample.index:
            features[i] = sample[probeName]
        else:
            features[i] = np.nan

    for i, feature in enumerate(features):
        if np.isnan(feature):
            features[i] = imputeMedians[i]

    features = np.clip(features, config.BETA_CLIP_MIN, config.BETA_CLIP_MAX)

    features = logitTransform(features)

    features = (features - featureMean) / featureStd

    return features

def predict(model, features, inverseEncoder, device):
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)

    topProbs, topIndices = torch.topk(probabilities, k=min(3, len(probabilities)))

    predictions = []
    for prob, idx in zip(topProbs.cpu().numpy(), topIndices.cpu().numpy()):
        predictions.append({
            'cancer_type': inverseEncoder[int(idx)],
            'probability': float(prob)
        })

    return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict cancer type from methylation data')
    parser.add_argument('input_file', type=str, help='Path to TSV/CSV file with methylation beta values')
    args = parser.parse_args()

    print("=" * 60)
    print("Cancer Type Classifier - Inference")
    print("=" * 60)

    model, probeNames, imputeMedians, featureMean, featureStd, inverseEncoder, device = loadArtifacts()

    print(f"\nLoading sample from: {args.input_file}")
    features = preprocessSample(args.input_file, probeNames, imputeMedians, featureMean, featureStd)

    print("\nMaking prediction...")
    predictions = predict(model, features, inverseEncoder, device)

    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)

    print(f"\nPredicted Cancer Type: {predictions[0]['cancer_type']}")
    print(f"Confidence: {predictions[0]['probability']:.4f}")

    print("\nTop 3 Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred['cancer_type']}: {pred['probability']:.4f}")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
