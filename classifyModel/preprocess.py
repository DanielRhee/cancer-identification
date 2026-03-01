import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import urllib.request
import gzip
import shutil

import config
from utils import logitTransform

def downloadManifests():
    manifestDir = config.ARTIFACTS_DIR / 'manifests'
    manifestDir.mkdir(exist_ok=True)

    sexChrProbes = set()

    for name, url in config.MANIFEST_URLS.items():
        try:
            print(f"Downloading {name} manifest...")
            gzPath = manifestDir / f'{name}.manifest.tsv.gz'
            tsvPath = manifestDir / f'{name}.manifest.tsv'

            urllib.request.urlretrieve(url, gzPath)

            with gzip.open(gzPath, 'rb') as fIn:
                with open(tsvPath, 'wb') as fOut:
                    shutil.copyfileobj(fIn, fOut)

            manifest = pd.read_csv(tsvPath, sep='\t', usecols=['Probe_ID', 'CpG_chrm'])
            sexProbes = manifest[manifest['CpG_chrm'].isin(['chrX', 'chrY'])]['Probe_ID'].tolist()
            sexChrProbes.update(sexProbes)
            print(f"  Found {len(sexProbes)} sex chromosome probes in {name}")

        except Exception as e:
            print(f"  Warning: Failed to download {name} manifest: {e}")

    return sexChrProbes

def loadCDR():
    print("Loading CDR data...")
    cdr = pd.read_excel(config.CDR_FILE, sheet_name='TCGA-CDR')

    cdr = cdr[['bcr_patient_barcode', 'type']].dropna()
    cdr = cdr.rename(columns={'bcr_patient_barcode': 'patient_id', 'type': 'cancer_type'})

    print(f"  Loaded {len(cdr)} patients with {cdr['cancer_type'].nunique()} cancer types")
    return cdr

def filterSamples(methylationHeader, cdr):
    print("Filtering samples...")

    sampleBarcodes = [col.strip('"') for col in methylationHeader[1:]]

    primaryTumors = []
    patientMap = {}

    for barcode in sampleBarcodes:
        parts = barcode.split('-')
        if len(parts) >= 4:
            sampleType = parts[3][:2]
            if sampleType == '01':
                patientId = '-'.join(parts[:3])

                if patientId in cdr['patient_id'].values:
                    if patientId not in patientMap:
                        patientMap[patientId] = barcode
                        primaryTumors.append(barcode)

    print(f"  Kept {len(primaryTumors)} primary tumor samples from {len(patientMap)} unique patients")
    return primaryTumors, patientMap

def loadMethylationData(keptSamples):
    print("Loading methylation data...")

    useCols = ['"Composite Element REF"'] + [f'"{sample}"' for sample in keptSamples]

    methylation = pd.read_csv(
        config.METHYLATION_FILE,
        sep='\t',
        usecols=useCols,
        dtype={f'"{col}"': 'float32' for col in keptSamples},
        low_memory=False
    )

    methylation = methylation.rename(columns={'"Composite Element REF"': 'Composite Element REF'})
    methylation.columns = [col.strip('"') for col in methylation.columns]
    methylation = methylation.set_index('Composite Element REF')
    methylation = methylation.T

    print(f"  Loaded {methylation.shape[0]} samples x {methylation.shape[1]} probes")
    return methylation

def filterProbes(methylation):
    print("Filtering probes...")

    sexChrProbes = downloadManifests()

    if sexChrProbes:
        probesBeforeSex = methylation.shape[1]
        methylation = methylation.loc[:, ~methylation.columns.isin(sexChrProbes)]
        print(f"  Removed {probesBeforeSex - methylation.shape[1]} sex chromosome probes")
    else:
        print("  Skipping sex chromosome filtering (no manifests available)")

    missingnessRate = methylation.isnull().sum() / len(methylation)
    missingnessRate = methylation.isnull().sum() / len(methylation)
    keptProbes = missingnessRate[missingnessRate <= config.MISSINGNESS_THRESHOLD].index
    methylation = methylation[keptProbes]
    print(f"  Removed {len(missingnessRate) - len(keptProbes)} probes with >{config.MISSINGNESS_THRESHOLD*100}% missingness")

    return methylation

def preprocessData(methylation, cdr, patientMap):
    print("Preprocessing data...")

    sampleToPatient = {barcode: patient for patient, barcode in patientMap.items()}
    methylation['patient_id'] = methylation.index.map(sampleToPatient)

    data = methylation.merge(cdr, on='patient_id', how='left')
    data = data.drop(columns=['patient_id'])

    cancerTypes = sorted(data['cancer_type'].unique())
    labelEncoder = {ct: idx for idx, ct in enumerate(cancerTypes)}
    data['label'] = data['cancer_type'].map(labelEncoder)

    labels = data['label'].values
    features = data.drop(columns=['cancer_type', 'label']).values

    print(f"  Final dataset: {features.shape[0]} samples x {features.shape[1]} features")
    print(f"  Cancer types: {len(cancerTypes)}")

    trainIdx, tempIdx = train_test_split(
        np.arange(len(features)),
        test_size=(1 - config.TRAIN_RATIO),
        stratify=labels,
        random_state=config.RANDOM_SEED
    )

    valTestRatio = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    valIdx, testIdx = train_test_split(
        tempIdx,
        test_size=(1 - valTestRatio),
        stratify=labels[tempIdx],
        random_state=config.RANDOM_SEED
    )

    print(f"  Train: {len(trainIdx)} | Val: {len(valIdx)} | Test: {len(testIdx)}")

    trainFeatures, trainLabels = features[trainIdx], labels[trainIdx]
    valFeatures, valLabels = features[valIdx], labels[valIdx]
    testFeatures, testLabels = features[testIdx], labels[testIdx]

    trainVariances = np.var(trainFeatures, axis=0)
    topIndices = np.argsort(trainVariances)[-config.NUM_PROBES:]
    
    trainFeatures = trainFeatures[:, topIndices]
    valFeatures = valFeatures[:, topIndices]
    testFeatures = testFeatures[:, topIndices]
    
    probeNames = np.array(data.drop(columns=['cancer_type', 'label']).columns)[topIndices].tolist()
    print(f"  Selected top {config.NUM_PROBES} most variable probes based on training set")

    imputeMedians = np.nanmedian(trainFeatures, axis=0)

    for i in range(trainFeatures.shape[1]):
        trainFeatures[np.isnan(trainFeatures[:, i]), i] = imputeMedians[i]
        valFeatures[np.isnan(valFeatures[:, i]), i] = imputeMedians[i]
        testFeatures[np.isnan(testFeatures[:, i]), i] = imputeMedians[i]

    trainFeatures = np.clip(trainFeatures, config.BETA_CLIP_MIN, config.BETA_CLIP_MAX)
    valFeatures = np.clip(valFeatures, config.BETA_CLIP_MIN, config.BETA_CLIP_MAX)
    testFeatures = np.clip(testFeatures, config.BETA_CLIP_MIN, config.BETA_CLIP_MAX)

    trainFeatures = logitTransform(trainFeatures)
    valFeatures = logitTransform(valFeatures)
    testFeatures = logitTransform(testFeatures)

    featureMean = trainFeatures.mean(axis=0)
    featureStd = trainFeatures.std(axis=0)

    trainFeatures = (trainFeatures - featureMean) / featureStd
    valFeatures = (valFeatures - featureMean) / featureStd
    testFeatures = (testFeatures - featureMean) / featureStd

    classDistribution = pd.Series(trainLabels).value_counts()
    classWeights = 1.0 / classDistribution
    classWeights = classWeights / classWeights.sum() * len(classWeights)
    classWeights = classWeights.sort_index().values

    return {
        'train_data': torch.tensor(trainFeatures, dtype=torch.float32),
        'val_data': torch.tensor(valFeatures, dtype=torch.float32),
        'test_data': torch.tensor(testFeatures, dtype=torch.float32),
        'train_labels': torch.tensor(trainLabels, dtype=torch.long),
        'val_labels': torch.tensor(valLabels, dtype=torch.long),
        'test_labels': torch.tensor(testLabels, dtype=torch.long),
        'label_encoder': labelEncoder,
        'probe_names': probeNames,
        'impute_medians': torch.tensor(imputeMedians, dtype=torch.float32),
        'feature_mean': torch.tensor(featureMean, dtype=torch.float32),
        'feature_std': torch.tensor(featureStd, dtype=torch.float32),
        'class_weights': torch.tensor(classWeights, dtype=torch.float32)
    }

def saveArtifacts(artifacts):
    print("Saving artifacts...")
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    for key in ['train_data', 'val_data', 'test_data', 'train_labels', 'val_labels',
                'test_labels', 'impute_medians', 'feature_mean', 'feature_std', 'class_weights']:
        torch.save(artifacts[key], config.getArtifactPath(key))

    with open(config.getArtifactPath('label_encoder'), 'w') as f:
        json.dump(artifacts['label_encoder'], f, indent=2)

    with open(config.getArtifactPath('probe_names'), 'w') as f:
        json.dump(artifacts['probe_names'], f, indent=2)

    meta = {
        'num_samples': {
            'train': int(len(artifacts['train_labels'])),
            'val': int(len(artifacts['val_labels'])),
            'test': int(len(artifacts['test_labels']))
        },
        'num_features': int(artifacts['train_data'].shape[1]),
        'num_classes': len(artifacts['label_encoder']),
        'cancer_types': {v: k for k, v in artifacts['label_encoder'].items()}
    }

    with open(config.getArtifactPath('preprocess_meta'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("  All artifacts saved successfully")
    return meta

def main():
    print("=" * 60)
    print("Cancer Type Classifier - Data Preprocessing")
    print("=" * 60)

    cdr = loadCDR()

    print("Reading methylation file header...")
    with open(config.METHYLATION_FILE, 'r') as f:
        header = f.readline().strip().split('\t')

    keptSamples, patientMap = filterSamples(header, cdr)

    methylation = loadMethylationData(keptSamples)

    methylation = filterProbes(methylation)

    artifacts = preprocessData(methylation, cdr, patientMap)

    meta = saveArtifacts(artifacts)

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Train samples: {meta['num_samples']['train']}")
    print(f"Val samples: {meta['num_samples']['val']}")
    print(f"Test samples: {meta['num_samples']['test']}")
    print(f"Features: {meta['num_features']}")
    print(f"Classes: {meta['num_classes']}")
    print("\nCancer type distribution (train set):")

    trainLabels = artifacts['train_labels'].numpy()
    for label, count in sorted(pd.Series(trainLabels).value_counts().items()):
        cancerType = meta['cancer_types'][label]
        print(f"  {cancerType}: {count}")

if __name__ == '__main__':
    main()
