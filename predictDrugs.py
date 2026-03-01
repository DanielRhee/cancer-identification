#!/usr/bin/env python3
"""
Drug Sensitivity Prediction CLI

Predicts cancer type and drug sensitivity from methylation data using the full 3-model pipeline:
  Model 1: Methylation → Cancer Type Classification
  Model 2: Methylation → Gene Expression Translation
  Model 3: Gene Expression → Drug IC50 Prediction
"""

import torch
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'classifyModel'))
sys.path.append(str(Path(__file__).parent / 'drugability'))

from classifyModel import config as classifyConfig
from classifyModel import infer as classifyInfer
from classifyModel.utils import getDevice

from drugability import config as drugConfig
from drugability.model import ExpressionTranslator, DrugPredictor

def loadAllModels():
    print("Program start")

    device = getDevice()
    classifyModel, probeNames, imputeMedians, featureMean, featureStd, cancerTypeEncoder, _ = classifyInfer.loadArtifacts()
    classifyModel = classifyModel.to(device)
    with open(drugConfig.getArtifactPath('gene_overlap'), 'r') as f:
        geneOverlap = json.load(f)

    expressionMean = torch.load(drugConfig.getArtifactPath('expression_mean')).to(device)
    expressionStd = torch.load(drugConfig.getArtifactPath('expression_std')).to(device)

    translatorModel = ExpressionTranslator(
        numGenes=len(geneOverlap),
        encoderDims=drugConfig.TRANSLATOR_ENCODER_DIMS,
        decoderDims=drugConfig.TRANSLATOR_DECODER_DIMS,
        dropout=drugConfig.TRANSLATOR_DROPOUT
    ).to(device)

    translatorModel.load_state_dict(torch.load(drugConfig.getArtifactPath('translator_best_model'), map_location=device, weights_only=True))
    translatorModel.eval()
    with open(drugConfig.getArtifactPath('drug_names'), 'r') as f:
        drugNames = json.load(f)

    predictorModel = DrugPredictor(
        numGenes=len(geneOverlap),
        numDrugs=len(drugNames),
        hiddenDims=drugConfig.PREDICTOR_HIDDEN_DIMS,
        dropout=drugConfig.PREDICTOR_DROPOUT
    ).to(device)

    predictorModel.load_state_dict(torch.load(drugConfig.getArtifactPath('predictor_best_model'), map_location=device, weights_only=True))
    predictorModel.eval()
    print("Loaded Models ")

    with open(drugConfig.getArtifactPath('cancer_type_drug_map'), 'r') as f:
        cancerTypeDrugMap = json.load(f)

    perDrugReliabilityPath = drugConfig.ARTIFACTS_DIR / 'perDrugReliability.json'
    if perDrugReliabilityPath.exists():
        with open(perDrugReliabilityPath, 'r') as f:
            perDrugReliability = json.load(f)
    else:
        print("  Warning: perDrugReliability.json not found, using default scores")
        perDrugReliability = {drug: 0.5 for drug in drugNames}

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

def loadExampleSample():
    """Load a random sample from the test set as an example."""
    print("Loading example from TCGA test set...")

    testData = torch.load(classifyConfig.getArtifactPath('test_data'), weights_only=False)
    testLabels = torch.load(classifyConfig.getArtifactPath('test_labels'), weights_only=False)

    with open(classifyConfig.getArtifactPath('label_encoder'), 'r') as f:
        labelEncoder = json.load(f)
    inverseEncoder = {v: k for k, v in labelEncoder.items()}

    with open(classifyConfig.getArtifactPath('probe_names'), 'r') as f:
        probeNames = json.load(f)

    idx = np.random.randint(0, len(testData))

    sampleData = testData[idx].numpy()
    trueLabel = inverseEncoder[int(testLabels[idx])]

    sampleDf = pd.Series(sampleData, index=probeNames)

    print("  Selected test sample: " + str(idx))
    print("  True cancer type: " + trueLabel)
    print()

    return sampleDf, trueLabel

def predictFromMethylation(methylationData, artifacts, topK=10, showAll=False):
    """Run full 3-model pipeline on methylation data."""
    device = artifacts['device']
    
    print("RUNNING PREDICTION PIPELINE")

    print("Preprocessing methylation data")
    processedMethylation = classifyInfer.preprocessSample(
        methylationData,
        artifacts['probe_names'],
        artifacts['impute_medians'],
        artifacts['feature_mean'],
        artifacts['feature_std']
    )
    methylationTensor = torch.tensor(processedMethylation, dtype=torch.float32).unsqueeze(0).to(device)

    print(" Predicting cancer type")
    with torch.no_grad():
        cancerLogits = artifacts['classify_model'](methylationTensor)
        cancerProbs = torch.softmax(cancerLogits, dim=1)

        topProbs, topIndices = torch.topk(cancerProbs[0], k=3)

    predictedCancerType = artifacts['cancer_type_encoder'][topIndices[0].item()]
    predictedConfidence = topProbs[0].item()

    print("Predicted: " + predictedCancerType + " (" + "{:.1%}".format(predictedConfidence) + " confidence)")
    print("Top 3 predictions:")
    for i in range(3):
        cancerType = artifacts['cancer_type_encoder'][topIndices[i].item()]
        confidence = topProbs[i].item()
        print(str(i+1) + ". " + cancerType + " - " + "{:.1%}".format(confidence))

    print("Translating methylation to gene expression")
    with torch.no_grad():
        predExpression = artifacts['translator_model'](methylationTensor)
        predExpression = predExpression * artifacts['expression_std'] + artifacts['expression_mean']
        
    print("Predicting Drug Sensativity")
    with torch.no_grad():
        predExpressionNorm = (predExpression - predExpression.mean(dim=1, keepdim=True)) / (predExpression.std(dim=1, keepdim=True) + 1e-6)
        predIc50 = artifacts['predictor_model'](predExpressionNorm)

    predIc50 = predIc50.cpu().numpy()[0]

    relevantDrugs = artifacts['cancer_type_drug_map'].get(predictedCancerType, [])

    #if len(relevantDrugs) == 0:
    #    print("  No specific drugs found for " + predictedCancerType)
    #    print("  Using all " + str(len(artifacts['drug_names'])) + " drugs")
    #    relevantDrugs = artifacts['drug_names']
    #else:
    #    print("  Found " + str(len(relevantDrugs)) + " drugs tested on " + predictedCancerType + " cell lines")
    #print()

    drugScores = []
    for i, drugName in enumerate(artifacts['drug_names']):
        if drugName in relevantDrugs or len(relevantDrugs) == 0:
            reliability = artifacts['per_drug_reliability'].get(drugName, 0.0)
            drugScores.append({
                'drug': drugName,
                'predicted_ic50': float(predIc50[i]),
                'reliability': reliability
            })

    drugScores.sort(key=lambda x: x['predicted_ic50'])

    print()
    print("------ RESULTS ------")
    print("Predicted Cancer Type: " + predictedCancerType)
    print("Confidence: " + "{:.1%}".format(predictedConfidence))

    if showAll:
        displayCount = len(drugScores)
        print("All " + str(displayCount) + " Cancer-Relevant Drug Predictions:")
    else:
        displayCount = min(topK, len(drugScores))
        print("Top " + str(displayCount) + " Most Sensitive Drugs:")

    for rank, drug in enumerate(drugScores[:displayCount], 1):
        if drug['reliability'] > 0.5:
            reliabilityLevel = "High"
        elif drug['reliability'] > 0.3:
            reliabilityLevel = "Medium"
        else:
            reliabilityLevel = "Low"

        output = str(rank) + ". " + drug['drug'] + ": IC50=" + "{:.3f}".format(drug['predicted_ic50']) + " Reliability=" + reliabilityLevel + " (" + "{:.3f}".format(drug['reliability']) + ")"
        print(output)
    return {
        'cancer_type': predictedCancerType,
        'cancer_confidence': predictedConfidence,
        'top_3_predictions': [
            {'type': artifacts['cancer_type_encoder'][topIndices[i].item()],
             'confidence': float(topProbs[i])}
            for i in range(3)
        ],
        'top_drugs': drugScores[:topK],
        'all_filtered_drugs': drugScores
    }

def main():
    parser = argparse.ArgumentParser(
        description='Predict cancer type and drug sensitivity from methylation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,

        epilog="""
Examples:
  # Run on example from test set
  python predictDrugs.py --example

  # Run on custom methylation file
  python predictDrugs.py --input my_methylation.tsv

  # Show top 20 drugs
  python predictDrugs.py --example --top-k 20

  # Save results to JSON
  python predictDrugs.py --example --output results.json

  # Show all cancer-relevant drugs
  python predictDrugs.py --example --show-all
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--example', action='store_true',
                       help='Run on random example from TCGA test set')
    group.add_argument('--input', type=str,
                       help='Path to methylation data file (TSV or CSV)')

    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top drug recommendations to display (default: 10)')
    parser.add_argument('--output', type=str,
                        help='Path to save results as JSON')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all cancer-relevant drugs instead of just top-K')

    args = parser.parse_args()

    artifacts = loadAllModels()

    if args.example:
        methylationData, trueLabel = loadExampleSample()
    else:
        inputPath = Path(args.input)
        if inputPath.suffix == '.tsv':
            methylationData = pd.read_csv(inputPath, sep='\t', index_col=0).iloc[:, 0]
        else:
            methylationData = pd.read_csv(inputPath, index_col=0).iloc[:, 0]
        trueLabel = None

    results = predictFromMethylation(methylationData, artifacts, topK=args.top_k, showAll=args.show_all)

    if trueLabel:
        print()
        print("Ground Truth: " + trueLabel)
        if results['cancer_type'] == trueLabel:
            print("Prediction CORRECT")
        else:
            print("Prediction incorrect (predicted " + results['cancer_type'] + ")")
        print()

    if args.output:
        outputData = results.copy()
        if trueLabel:
            outputData['true_cancer_type'] = trueLabel

        with open(args.output, 'w') as f:
            json.dump(outputData, f, indent=2)
        print("Results saved to " + args.output)
        print()

if __name__ == '__main__':
    main()
