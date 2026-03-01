#!/usr/bin/env python3
import sys
import argparse
import torch
import json
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'classifyModel'))

from classifyModel import config
from classifyModel.utils import getDevice
from classifyModel.model import CancerClassifier

class CancerClassifierCLI:
    def __init__(self):
        self.loadModel()
        self.loadTestData()

    def loadModel(self):
        print("Loading model and preprocessing parameters...")

        with open(config.getArtifactPath('label_encoder'), 'r') as f:
            self.labelEncoder = json.load(f)

        self.inverseEncoder = {v: k for k, v in self.labelEncoder.items()}

        with open(config.getArtifactPath('probe_names'), 'r') as f:
            self.probeNames = json.load(f)

        inputDim = len(self.probeNames)
        numClasses = len(self.labelEncoder)

        self.device = getDevice()
        self.model = CancerClassifier(inputDim, numClasses).to(self.device)

        checkpoint = torch.load(config.getArtifactPath('best_model'), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded (Val F1: {checkpoint['val_f1']:.4f}, Device: {self.device})")

    def loadTestData(self):
        self.testData = torch.load(config.getArtifactPath('test_data'), weights_only=False)
        self.testLabels = torch.load(config.getArtifactPath('test_labels'), weights_only=False)
        print(f"✓ Loaded {len(self.testLabels)} test samples")

    def predict(self, features, showTop=5):
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        features = features.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)

        topProbs, topIndices = torch.topk(probabilities, k=min(showTop, len(probabilities)))

        predictions = []
        for prob, idx in zip(topProbs.cpu().numpy(), topIndices.cpu().numpy()):
            predictions.append({
                'cancerType': self.inverseEncoder[int(idx)],
                'probability': float(prob),
                'label': int(idx)
            })

        return predictions

    def classifyTestSample(self, sampleIdx):
        if sampleIdx < 0 or sampleIdx >= len(self.testLabels):
            print(f"Error: Sample index must be between 0 and {len(self.testLabels)-1}")
            return

        features = self.testData[sampleIdx]
        trueLabel = int(self.testLabels[sampleIdx])
        trueCancerType = self.inverseEncoder[trueLabel]

        print("\n" + "="*70)
        print(f"TEST SAMPLE #{sampleIdx}")
        print("="*70)
        print(f"True Cancer Type: {trueCancerType}")
        print("-"*70)

        predictions = self.predict(features, showTop=5)

        print("\nTop 5 Predictions:")
        for i, pred in enumerate(predictions, 1):
            isCorrect = pred['cancerType'] == trueCancerType
            marker = "✓" if isCorrect else " "
            bar = "█" * int(pred['probability'] * 50)
            print(f"{marker} {i}. {pred['cancerType']:<25} {pred['probability']:6.2%} {bar}")

        if predictions[0]['cancerType'] == trueCancerType:
            print("\n✓ CORRECT PREDICTION")
        else:
            print("\n✗ INCORRECT PREDICTION")
        print("="*70)

    def showRandomExamples(self, numExamples=5):
        print("\n" + "="*70)
        print(f"RANDOM TEST EXAMPLES (showing {numExamples})")
        print("="*70)

        indices = random.sample(range(len(self.testLabels)), min(numExamples, len(self.testLabels)))

        correct = 0
        for idx in indices:
            features = self.testData[idx]
            trueLabel = int(self.testLabels[idx])
            trueCancerType = self.inverseEncoder[trueLabel]

            predictions = self.predict(features, showTop=3)
            predictedType = predictions[0]['cancerType']
            confidence = predictions[0]['probability']

            isCorrect = predictedType == trueCancerType
            if isCorrect:
                correct += 1

            marker = "✓" if isCorrect else "✗"
            print(f"\n{marker} Sample #{idx:4d}")
            print(f"  True:      {trueCancerType}")
            print(f"  Predicted: {predictedType} ({confidence:.2%})")
            if not isCorrect:
                print(f"  Top 3: {', '.join([p['cancerType'] for p in predictions[:3]])}")

        print("\n" + "-"*70)
        print(f"Accuracy: {correct}/{len(indices)} ({100*correct/len(indices):.1f}%)")
        print("="*70)

    def listCancerTypes(self):
        print("\n" + "="*70)
        print("AVAILABLE CANCER TYPES")
        print("="*70)

        cancerTypes = sorted(self.labelEncoder.keys())
        for i, cancerType in enumerate(cancerTypes, 1):
            label = self.labelEncoder[cancerType]
            count = (self.testLabels == label).sum().item()
            print(f"{i:2d}. {cancerType:<30} (Label: {label}, Test samples: {count})")

        print("="*70)

    def showStats(self):
        print("\n" + "="*70)
        print("MODEL STATISTICS")
        print("="*70)

        with open(config.getArtifactPath('preprocess_meta'), 'r') as f:
            meta = json.load(f)

        print(f"\nDataset Split:")
        print(f"  Train:    {meta['num_samples']['train']:,} samples")
        print(f"  Val:      {meta['num_samples']['val']:,} samples")
        print(f"  Test:     {meta['num_samples']['test']:,} samples")
        print(f"\nModel Configuration:")
        print(f"  Features: {meta['num_features']:,} probes")
        print(f"  Classes:  {meta['num_classes']} cancer types")
        print(f"  Hidden:   {config.HIDDEN_DIMS}")

        totalParams = sum(p.numel() for p in self.model.parameters())
        print(f"  Params:   {totalParams:,}")

        print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Cancer Type Classification CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --example 42                 # Classify test sample #42
  %(prog)s --random 10                  # Show 10 random test examples
  %(prog)s --list                       # List all cancer types
  %(prog)s --stats                      # Show model statistics
        """
    )

    parser.add_argument('--example', '-e', type=int, metavar='N',
                        help='Classify test sample by index (0 to num_samples-1)')
    parser.add_argument('--random', '-r', type=int, metavar='N', default=None,
                        help='Show N random test examples')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available cancer types')
    parser.add_argument('--stats', '-s', action='store_true',
                        help='Show model statistics')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    print("\n" + "="*70)
    print("CANCER TYPE CLASSIFIER")
    print("="*70 + "\n")

    cli = CancerClassifierCLI()

    if args.list:
        cli.listCancerTypes()

    if args.stats:
        cli.showStats()

    if args.example is not None:
        cli.classifyTestSample(args.example)

    if args.random is not None:
        cli.showRandomExamples(args.random)

if __name__ == '__main__':
    main()
