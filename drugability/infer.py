import torch
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import sys

parentDir = Path(__file__).parent.parent
sys.path.append(str(parentDir))
sys.path.append(str(parentDir / 'classifyModel'))

from classifyModel import infer as classifyInfer
from classifyModel.utils import getDevice

import config
from model import ExpressionTranslator, DrugPredictor

def loadArtifacts():
    print("Loading all models and preprocessing parameters...")
    print()

    print("  [1/3] Loading cancer type classifier (Model 1)...")
    classifyModel, probeNames, imputeMedians, featureMean, featureStd, cancerTypeEncoder, device = classifyInfer.loadArtifacts()

    print("  [2/3] Loading expression translator (Model 2)...")
    with open(config.getArtifactPath('gene_overlap'), 'r') as f:
        geneOverlap = json.load(f)

    expressionMean = torch.load(config.getArtifactPath('expression_mean')).to(device)
    expressionStd = torch.load(config.getArtifactPath('expression_std')).to(device)

    translatorModel = ExpressionTranslator(
        numGenes=len(geneOverlap),
        encoderDims=config.TRANSLATOR_ENCODER_DIMS,
        decoderDims=config.TRANSLATOR_DECODER_DIMS,
        dropout=config.TRANSLATOR_DROPOUT
    ).to(device)

    translatorModel.load_state_dict(torch.load(config.getArtifactPath('translator_best_model'), map_location=device, weights_only=True))
    translatorModel.eval()

    print("  [3/3] Loading drug predictor (Model 3)...")
    with open(config.getArtifactPath('drug_names'), 'r') as f:
        drugNames = json.load(f)

    predictorModel = DrugPredictor(
        numGenes=len(geneOverlap),
        numDrugs=len(drugNames),
        hiddenDims=config.PREDICTOR_HIDDEN_DIMS,
        dropout=config.PREDICTOR_DROPOUT
    ).to(device)

    predictorModel.load_state_dict(torch.load(config.getArtifactPath('predictor_best_model'), map_location=device, weights_only=True))
    predictorModel.eval()

    with open(config.getArtifactPath('cancer_type_drug_map'), 'r') as f:
        cancerTypeDrugMap = json.load(f)

    with open(config.ARTIFACTS_DIR / 'perDrugReliability.json', 'r') as f:
        perDrugReliability = json.load(f)

    print()
    print(f"  All models loaded successfully!")
    print(f"  Device: {device}")
    print(f"  Genes: {len(geneOverlap)}")
    print(f"  Drugs: {len(drugNames)}")
    print()

    return {
        'classify_model': classifyModel,
        'translator_model': translatorModel,
        'predictor_model': predictorModel,
        'probe_names': probeNames,
        'impute_medians': imputeMedians,
        'feature_mean': featureMean,
        'feature_std': featureStd,
        'expression_mean': expressionMean,
        'expression_std': expressionStd,
        'cancer_type_encoder': cancerTypeEncoder,
        'gene_overlap': geneOverlap,
        'drug_names': drugNames,
        'cancer_type_drug_map': cancerTypeDrugMap,
        'per_drug_reliability': perDrugReliability,
        'device': device
    }

def predictDrugSensitivity(methylationData, artifacts, topK=10):
    device = artifacts['device']

    print("=" * 70)
    print("CANCER TYPE & DRUG SENSITIVITY PREDICTION PIPELINE")
    print("=" * 70)
    print()

    print("Step 1: Preprocessing methylation data...")
    processedMethylation = classifyInfer.preprocessSample(
        methylationData,
        artifacts['probe_names'],
        artifacts['impute_medians'],
        artifacts['feature_mean'],
        artifacts['feature_std']
    )
    methylationTensor = torch.tensor(processedMethylation, dtype=torch.float32).unsqueeze(0).to(device)

    print("Step 2: Predicting cancer type (Model 1)...")
    with torch.no_grad():
        cancerLogits = artifacts['classify_model'](methylationTensor)
        cancerProbs = torch.softmax(cancerLogits, dim=1)
        cancerPredIdx = cancerProbs.argmax(dim=1).item()
        cancerConfidence = cancerProbs[0, cancerPredIdx].item()

    predictedCancerType = artifacts['cancer_type_encoder'][cancerPredIdx]

    print(f"  Predicted cancer type: {predictedCancerType}")
    print(f"  Confidence: {cancerConfidence:.2%}")
    print()

    print("Step 3: Translating methylation to gene expression (Model 2)...")
    with torch.no_grad():
        predExpression = artifacts['translator_model'](methylationTensor)

        predExpression = predExpression * artifacts['expression_std'] + artifacts['expression_mean']

    print(f"  Predicted expression for {len(artifacts['gene_overlap'])} genes")
    print()

    print("Step 4: Predicting drug sensitivity (Model 3)...")
    with torch.no_grad():
        predExpression = (predExpression - predExpression.mean(dim=1, keepdim=True)) / (predExpression.std(dim=1, keepdim=True) + 1e-6)

        predIc50 = artifacts['predictor_model'](predExpression)

    predIc50 = predIc50.cpu().numpy()[0]

    print(f"  Predicted IC50 for {len(artifacts['drug_names'])} drugs")
    print()

    print("Step 5: Filtering drugs by cancer type...")
    relevantDrugs = artifacts['cancer_type_drug_map'].get(predictedCancerType, [])

    if len(relevantDrugs) == 0:
        print(f"  Warning: No cancer-specific drugs found for {predictedCancerType}")
        print(f"  Returning all {len(artifacts['drug_names'])} drugs")
        relevantDrugs = artifacts['drug_names']

    drugScores = []
    for i, drugName in enumerate(artifacts['drug_names']):
        if drugName in relevantDrugs:
            reliability = artifacts['per_drug_reliability'].get(drugName, 0.0)
            drugScores.append({
                'drug': drugName,
                'predicted_ic50': float(predIc50[i]),
                'reliability': reliability
            })

    drugScores.sort(key=lambda x: x['predicted_ic50'])

    print(f"  Filtered to {len(drugScores)} cancer-relevant drugs")
    print()

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Cancer Type: {predictedCancerType} ({cancerConfidence:.1%} confidence)")
    print()
    print(f"Top {topK} Drug Recommendations (by predicted sensitivity):")
    print()
    print(f"{'Rank':<6} {'Drug Name':<40} {'IC50':<12} {'Reliability':<12}")
    print("-" * 70)

    for rank, drug in enumerate(drugScores[:topK], 1):
        reliabilityFlag = "High" if drug['reliability'] > 0.5 else "Medium" if drug['reliability'] > 0.3 else "Low"
        print(f"{rank:<6} {drug['drug']:<40} {drug['predicted_ic50']:>8.3f}    {reliabilityFlag} ({drug['reliability']:.3f})")

    print()

    return {
        'cancer_type': predictedCancerType,
        'cancer_confidence': cancerConfidence,
        'top_drugs': drugScores[:topK],
        'all_filtered_drugs': drugScores
    }

def main():
    parser = argparse.ArgumentParser(description='Predict cancer type and drug sensitivity from methylation data')
    parser.add_argument('input_file', type=str, help='Path to methylation data file (TSV or CSV)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top drug recommendations to display (default: 10)')
    parser.add_argument('--output', type=str, help='Path to save results as JSON')

    args = parser.parse_args()

    artifacts = loadArtifacts()

    results = predictDrugSensitivity(args.input_file, artifacts, topK=args.top_k)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
