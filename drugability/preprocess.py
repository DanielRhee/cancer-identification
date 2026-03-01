import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

parentDir = Path(__file__).parent.parent
sys.path.append(str(parentDir))
sys.path.append(str(parentDir / 'classifyModel'))

from classifyModel.preprocess import loadCDR, filterSamples
from classifyModel import config as classifyConfig

import config

def recoverSplit():
    print("=" * 60)
    print("Recovering Model 1 train/val/test split...")
    print("=" * 60)

    cdr = loadCDR()

    print("Reading methylation file header...")
    with open(config.METHYLATION_FILE, 'r') as f:
        header = f.readline().strip().split('\t')

    keptSamples, patientMap = filterSamples(header, cdr)

    sampleToPatient = {barcode: patient for patient, barcode in patientMap.items()}

    patients = [sampleToPatient[sample] for sample in keptSamples]
    patientsCdr = cdr[cdr['patient_id'].isin(patients)].copy()
    patientsCdr = patientsCdr.drop_duplicates(subset=['patient_id'])
    patientsCdr = patientsCdr.set_index('patient_id')

    orderedPatients = [sampleToPatient[sample] for sample in keptSamples]
    cancerTypes = [patientsCdr.loc[p, 'cancer_type'] for p in orderedPatients]

    sortedCancerTypes = sorted(set(cancerTypes))
    labelEncoder = {ct: idx for idx, ct in enumerate(sortedCancerTypes)}
    labels = np.array([labelEncoder[ct] for ct in cancerTypes])

    trainIdx, tempIdx = train_test_split(
        np.arange(len(orderedPatients)),
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

    trainPatients = [orderedPatients[i] for i in trainIdx]
    valPatients = [orderedPatients[i] for i in valIdx]
    testPatients = [orderedPatients[i] for i in testIdx]

    print(f"  Train: {len(trainPatients)} patients")
    print(f"  Val: {len(valPatients)} patients")
    print(f"  Test: {len(testPatients)} patients")

    return {
        'train': set(trainPatients),
        'val': set(valPatients),
        'test': set(testPatients),
        'all': orderedPatients,
        'patient_map': patientMap
    }

def loadRNASeq(split):
    print("\n" + "=" * 60)
    print("Loading RNA-seq data...")
    print("=" * 60)

    print("Reading RNA-seq header...")
    with open(config.RNASEQ_FILE, 'r') as f:
        header = f.readline().strip().split('\t')

    sampleBarcodes = [col.strip('"') for col in header[1:]]

    primaryTumorSamples = []
    sampleToPatient = {}

    for barcode in sampleBarcodes:
        parts = barcode.split('-')
        if len(parts) >= 4:
            sampleType = parts[3][:2]
            if sampleType == '01':
                patientId = '-'.join(parts[:3])
                if patientId in split['all']:
                    primaryTumorSamples.append(barcode)
                    sampleToPatient[barcode] = patientId

    print(f"  Found {len(primaryTumorSamples)} primary tumor samples in RNA-seq data")

    patientsWithRna = set(sampleToPatient.values())

    trainPatients = list(split['train'] & patientsWithRna)
    valPatients = list(split['val'] & patientsWithRna)
    testPatients = list(split['test'] & patientsWithRna)

    print(f"  Patients with both methylation and RNA-seq:")
    print(f"    Train: {len(trainPatients)}")
    print(f"    Val: {len(valPatients)}")
    print(f"    Test: {len(testPatients)}")
    print(f"    Total: {len(patientsWithRna)}")

    patientToSample = {patient: sample for sample, patient in sampleToPatient.items()}
    selectedSamples = [patientToSample[p] for p in trainPatients + valPatients + testPatients]

    print(f"  Loading RNA-seq expression data for {len(selectedSamples)} samples...")
    useCols = ['gene_id'] + selectedSamples

    rnaseq = pd.read_csv(
        config.RNASEQ_FILE,
        sep='\t',
        usecols=useCols,
        low_memory=False
    )

    print(f"  Parsing gene symbols...")
    geneSymbols = []
    for geneId in rnaseq['gene_id']:
        if '|' in str(geneId):
            symbol = str(geneId).split('|')[0]
            if symbol != '?':
                geneSymbols.append(symbol)
            else:
                geneSymbols.append(None)
        else:
            geneSymbols.append(None)

    rnaseq['gene_symbol'] = geneSymbols
    rnaseq = rnaseq[rnaseq['gene_symbol'].notna()]
    rnaseq = rnaseq.drop(columns=['gene_id'])

    rnaseq = rnaseq.groupby('gene_symbol').mean()

    rnaseq = rnaseq.T

    print(f"  RNA-seq data: {rnaseq.shape[0]} samples x {rnaseq.shape[1]} genes")

    return {
        'data': rnaseq,
        'train_patients': trainPatients,
        'val_patients': valPatients,
        'test_patients': testPatients,
        'patient_to_sample': patientToSample
    }

def computeGeneOverlap(tcgaGenes):
    print("\n" + "=" * 60)
    print("Computing gene overlap with GDSC...")
    print("=" * 60)

    print("Loading GDSC expression gene symbols...")
    gdscExpression = pd.read_csv(
        config.GDSC_EXPRESSION_FILE,
        sep='\t',
        usecols=['GENE_SYMBOLS']
    )

    gdscGenes = set(gdscExpression['GENE_SYMBOLS'].values)
    tcgaGenesSet = set(tcgaGenes)

    overlap = sorted(list(gdscGenes & tcgaGenesSet))

    print(f"  TCGA genes: {len(tcgaGenesSet)}")
    print(f"  GDSC genes: {len(gdscGenes)}")
    print(f"  Overlap: {len(overlap)}")

    return overlap

def prepareModel2Data(rnaseqData, geneOverlap):
    print("\n" + "=" * 60)
    print("Preparing Model 2 (Translator) training data...")
    print("=" * 60)

    rnaseq = rnaseqData['data']
    patientToSample = rnaseqData['patient_to_sample']

    rnaseq = rnaseq[geneOverlap]

    print("Loading preprocessed methylation data...")
    trainMeth = torch.load(classifyConfig.getArtifactPath('train_data'))
    valMeth = torch.load(classifyConfig.getArtifactPath('val_data'))
    testMeth = torch.load(classifyConfig.getArtifactPath('test_data'))
    trainLabels = torch.load(classifyConfig.getArtifactPath('train_labels'))
    valLabels = torch.load(classifyConfig.getArtifactPath('val_labels'))
    testLabels = torch.load(classifyConfig.getArtifactPath('test_labels'))

    with open(classifyConfig.getArtifactPath('preprocess_meta'), 'r') as f:
        meta = json.load(f)

    with open(classifyConfig.getArtifactPath('label_encoder'), 'r') as f:
        labelEncoder = json.load(f)

    inverseEncoder = {v: k for k, v in labelEncoder.items()}

    cdr = loadCDR()
    cdr = cdr.set_index('patient_id')

    print("Reading methylation header to get original patient order...")
    with open(config.METHYLATION_FILE, 'r') as f:
        header = f.readline().strip().split('\t')

    keptSamples, patientMap = filterSamples(header, cdr.reset_index())
    sampleToPatient = {barcode: patient for patient, barcode in patientMap.items()}
    orderedPatients = [sampleToPatient[sample] for sample in keptSamples]

    cancerTypes = [cdr.loc[p, 'cancer_type'] for p in orderedPatients]
    sortedCancerTypes = sorted(set(cancerTypes))
    localLabelEncoder = {ct: idx for idx, ct in enumerate(sortedCancerTypes)}
    labels = np.array([localLabelEncoder[ct] for ct in cancerTypes])

    trainIdx, tempIdx = train_test_split(
        np.arange(len(orderedPatients)),
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

    trainPatients = [orderedPatients[i] for i in trainIdx]
    valPatients = [orderedPatients[i] for i in valIdx]
    testPatients = [orderedPatients[i] for i in testIdx]

    def matchPatientsToMethylation(splitPatients, methTensor, splitLabels):
        patientsWithRna = [p for p in splitPatients if p in patientToSample]
        rnaSamples = [patientToSample[p] for p in patientsWithRna]

        indices = [i for i, p in enumerate(splitPatients) if p in patientToSample]

        methSubset = methTensor[indices]

        rnaSubset = rnaseq.loc[rnaSamples].values

        return methSubset, torch.tensor(rnaSubset, dtype=torch.float32), len(patientsWithRna)

    trainMethMatched, trainRna, nTrain = matchPatientsToMethylation(trainPatients, trainMeth, trainLabels)
    valMethMatched, valRna, nVal = matchPatientsToMethylation(valPatients, valMeth, valLabels)
    testMethMatched, testRna, nTest = matchPatientsToMethylation(testPatients, testMeth, testLabels)

    print(f"  Matched samples:")
    print(f"    Train: {nTrain}")
    print(f"    Val: {nVal}")
    print(f"    Test: {nTest}")

    trainRna = torch.nan_to_num(trainRna, nan=0.0)
    valRna = torch.nan_to_num(valRna, nan=0.0)
    testRna = torch.nan_to_num(testRna, nan=0.0)

    expressionMean = trainRna.mean(dim=0)
    expressionStd = trainRna.std(dim=0)
    expressionStd = torch.where(expressionStd < 1e-6, torch.ones_like(expressionStd), expressionStd)

    trainRna = (trainRna - expressionMean) / expressionStd
    valRna = (valRna - expressionMean) / expressionStd
    testRna = (testRna - expressionMean) / expressionStd

    print(f"  Expression data normalized (z-score)")
    print(f"    Train shape: {trainRna.shape}")
    print(f"    Val shape: {valRna.shape}")
    print(f"    Test shape: {testRna.shape}")

    return {
        'train_methylation': trainMethMatched,
        'val_methylation': valMethMatched,
        'test_methylation': testMethMatched,
        'train_expression': trainRna,
        'val_expression': valRna,
        'test_expression': testRna,
        'expression_mean': expressionMean,
        'expression_std': expressionStd
    }

def prepareModel3Data(geneOverlap):
    print("\n" + "=" * 60)
    print("Preparing Model 3 (Predictor) training data...")
    print("=" * 60)

    print("Loading GDSC dose-response data...")
    doseResponse = pd.read_excel(config.GDSC_DOSE_RESPONSE_FILE)

    print(f"  Loaded {len(doseResponse)} dose-response measurements")

    print("Loading GDSC expression data...")
    gdscExpression = pd.read_csv(
        config.GDSC_EXPRESSION_FILE,
        sep='\t'
    )

    gdscExpression = gdscExpression.set_index('GENE_SYMBOLS')
    gdscExpression = gdscExpression.drop(columns=['GENE_title'])

    def parseCosmicId(col):
        cleaned = col.replace('DATA.', '')
        if '.' in cleaned:
            cleaned = cleaned.split('.')[0]
        return int(cleaned)

    gdscExpression.columns = [parseCosmicId(col) for col in gdscExpression.columns]

    gdscExpression = gdscExpression.T

    print(f"  GDSC expression: {gdscExpression.shape[0]} cell lines x {gdscExpression.shape[1]} genes")

    doseResponse = doseResponse[doseResponse['COSMIC_ID'].isin(gdscExpression.index)]

    print(f"  Filtered to {len(doseResponse)} measurements with expression data")

    ic50Matrix = doseResponse.pivot_table(
        index='COSMIC_ID',
        columns='DRUG_NAME',
        values='LN_IC50',
        aggfunc='mean'
    )

    print(f"  IC50 matrix: {ic50Matrix.shape[0]} cell lines x {ic50Matrix.shape[1]} drugs")

    coverage = ic50Matrix.notna().mean(axis=0)
    keptDrugs = coverage[coverage >= config.DRUG_COVERAGE_THRESHOLD].index.tolist()
    ic50Matrix = ic50Matrix[keptDrugs]

    print(f"  Kept {len(keptDrugs)} drugs with >{config.DRUG_COVERAGE_THRESHOLD*100}% coverage")

    mask = ic50Matrix.notna().astype(float)

    cellLines = ic50Matrix.index.tolist()

    trainLines, tempLines = train_test_split(
        cellLines,
        test_size=(1 - config.TRAIN_RATIO),
        random_state=config.RANDOM_SEED
    )

    valTestRatio = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    valLines, testLines = train_test_split(
        tempLines,
        test_size=(1 - valTestRatio),
        random_state=config.RANDOM_SEED
    )

    print(f"  Split: Train={len(trainLines)} | Val={len(valLines)} | Test={len(testLines)}")

    trainIc50 = ic50Matrix.loc[trainLines].values.copy()
    valIc50 = ic50Matrix.loc[valLines].values.copy()
    testIc50 = ic50Matrix.loc[testLines].values.copy()

    trainMask = mask.loc[trainLines].values
    valMask = mask.loc[valLines].values
    testMask = mask.loc[testLines].values

    drugMedians = np.nanmedian(trainIc50, axis=0)

    for i in range(trainIc50.shape[1]):
        trainIc50[np.isnan(trainIc50[:, i]), i] = drugMedians[i]
        valIc50[np.isnan(valIc50[:, i]), i] = drugMedians[i]
        testIc50[np.isnan(testIc50[:, i]), i] = drugMedians[i]

    ic50Mean = trainIc50.mean(axis=0)
    ic50Std = trainIc50.std(axis=0)
    ic50Std[ic50Std < 1e-6] = 1.0

    trainIc50 = (trainIc50 - ic50Mean) / ic50Std
    valIc50 = (valIc50 - ic50Mean) / ic50Std
    testIc50 = (testIc50 - ic50Mean) / ic50Std

    availableGenes = [g for g in geneOverlap if g in gdscExpression.columns]
    gdscExpressionFiltered = gdscExpression[availableGenes]

    trainExpr = gdscExpressionFiltered.loc[trainLines].values
    valExpr = gdscExpressionFiltered.loc[valLines].values
    testExpr = gdscExpressionFiltered.loc[testLines].values

    exprMean = trainExpr.mean(axis=0)
    exprStd = trainExpr.std(axis=0)
    exprStd[exprStd < 1e-6] = 1.0

    trainExpr = (trainExpr - exprMean) / exprStd
    valExpr = (valExpr - exprMean) / exprStd
    testExpr = (testExpr - exprMean) / exprStd

    print(f"  IC50 data normalized (z-score, imputed with train medians)")
    print(f"  Expression data normalized (z-score)")

    return {
        'train_expression': torch.tensor(trainExpr, dtype=torch.float32),
        'val_expression': torch.tensor(valExpr, dtype=torch.float32),
        'test_expression': torch.tensor(testExpr, dtype=torch.float32),
        'train_ic50': torch.tensor(trainIc50, dtype=torch.float32),
        'val_ic50': torch.tensor(valIc50, dtype=torch.float32),
        'test_ic50': torch.tensor(testIc50, dtype=torch.float32),
        'train_mask': torch.tensor(trainMask, dtype=torch.float32),
        'val_mask': torch.tensor(valMask, dtype=torch.float32),
        'test_mask': torch.tensor(testMask, dtype=torch.float32),
        'drug_names': keptDrugs
    }

def createCancerTypeDrugMap():
    print("\n" + "=" * 60)
    print("Creating cancer type to drug mapping...")
    print("=" * 60)

    print("Loading cell line details...")
    cellLineDetails = pd.read_excel(config.CELL_LINE_DETAILS_FILE)

    print("Loading dose-response data...")
    doseResponse = pd.read_excel(config.GDSC_DOSE_RESPONSE_FILE)

    print("Loading GDSC expression data for cell line filtering...")
    gdscExpression = pd.read_csv(
        config.GDSC_EXPRESSION_FILE,
        sep='\t'
    )
    gdscExpression = gdscExpression.set_index('GENE_SYMBOLS')
    gdscExpression = gdscExpression.drop(columns=['GENE_title'])

    def parseCosmicId(col):
        cleaned = col.replace('DATA.', '')
        if '.' in cleaned:
            cleaned = cleaned.split('.')[0]
        return int(cleaned)

    validCosmicIds = set([parseCosmicId(col) for col in gdscExpression.columns])

    cellLineDetails = cellLineDetails[cellLineDetails['COSMIC identifier'].isin(validCosmicIds)]
    doseResponse = doseResponse[doseResponse['COSMIC_ID'].isin(validCosmicIds)]

    tcgaColumn = 'Cancer Type\n(matching TCGA label)'

    cancerTypeDrugMap = {}

    for tcgaType in cellLineDetails[tcgaColumn].dropna().unique():
        cosmicIds = cellLineDetails[cellLineDetails[tcgaColumn] == tcgaType]['COSMIC identifier'].tolist()

        if len(cosmicIds) > 0:
            drugs = doseResponse[doseResponse['COSMIC_ID'].isin(cosmicIds)]['DRUG_NAME'].unique().tolist()

            if 'COADREAD' in tcgaType or tcgaType == 'COADREAD':
                cancerTypeDrugMap['COAD'] = drugs
                cancerTypeDrugMap['READ'] = drugs
            else:
                cancerTypeDrugMap[tcgaType] = drugs

    print(f"  Mapped {len(cancerTypeDrugMap)} TCGA cancer types to drugs")

    with open(classifyConfig.getArtifactPath('label_encoder'), 'r') as f:
        labelEncoder = json.load(f)

    allTcgaTypes = set(labelEncoder.keys())
    mappedTypes = set(cancerTypeDrugMap.keys())
    unmappedTypes = allTcgaTypes - mappedTypes

    if unmappedTypes:
        print(f"  Warning: {len(unmappedTypes)} cancer types have no GDSC mapping:")
        for ct in sorted(unmappedTypes):
            print(f"    - {ct}")
            cancerTypeDrugMap[ct] = []

    return cancerTypeDrugMap

def saveArtifacts(model2Data, model3Data, geneOverlap, cancerTypeDrugMap):
    print("\n" + "=" * 60)
    print("Saving artifacts...")
    print("=" * 60)

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    with open(config.getArtifactPath('gene_overlap'), 'w') as f:
        json.dump(geneOverlap, f, indent=2)

    torch.save(model2Data['train_methylation'], config.getArtifactPath('translator_train_methylation'))
    torch.save(model2Data['val_methylation'], config.getArtifactPath('translator_val_methylation'))
    torch.save(model2Data['test_methylation'], config.getArtifactPath('translator_test_methylation'))
    torch.save(model2Data['train_expression'], config.getArtifactPath('translator_train_expression'))
    torch.save(model2Data['val_expression'], config.getArtifactPath('translator_val_expression'))
    torch.save(model2Data['test_expression'], config.getArtifactPath('translator_test_expression'))
    torch.save(model2Data['expression_mean'], config.getArtifactPath('expression_mean'))
    torch.save(model2Data['expression_std'], config.getArtifactPath('expression_std'))

    torch.save(model3Data['train_expression'], config.getArtifactPath('predictor_train_expression'))
    torch.save(model3Data['val_expression'], config.getArtifactPath('predictor_val_expression'))
    torch.save(model3Data['test_expression'], config.getArtifactPath('predictor_test_expression'))
    torch.save(model3Data['train_ic50'], config.getArtifactPath('predictor_train_ic50'))
    torch.save(model3Data['val_ic50'], config.getArtifactPath('predictor_val_ic50'))
    torch.save(model3Data['test_ic50'], config.getArtifactPath('predictor_test_ic50'))
    torch.save(model3Data['train_mask'], config.getArtifactPath('predictor_train_mask'))
    torch.save(model3Data['val_mask'], config.getArtifactPath('predictor_val_mask'))
    torch.save(model3Data['test_mask'], config.getArtifactPath('predictor_test_mask'))

    with open(config.getArtifactPath('drug_names'), 'w') as f:
        json.dump(model3Data['drug_names'], f, indent=2)

    with open(config.getArtifactPath('cancer_type_drug_map'), 'w') as f:
        json.dump(cancerTypeDrugMap, f, indent=2)

    meta = {
        'num_genes': len(geneOverlap),
        'num_drugs': len(model3Data['drug_names']),
        'model2_samples': {
            'train': int(model2Data['train_methylation'].shape[0]),
            'val': int(model2Data['val_methylation'].shape[0]),
            'test': int(model2Data['test_methylation'].shape[0])
        },
        'model3_samples': {
            'train': int(model3Data['train_expression'].shape[0]),
            'val': int(model3Data['val_expression'].shape[0]),
            'test': int(model3Data['test_expression'].shape[0])
        }
    }

    with open(config.getArtifactPath('preprocess_meta'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("  All artifacts saved successfully")
    return meta

def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "DRUGABILITY PIPELINE - PREPROCESSING")
    print("=" * 80)

    split = recoverSplit()

    rnaseqData = loadRNASeq(split)

    geneOverlap = computeGeneOverlap(rnaseqData['data'].columns)

    model2Data = prepareModel2Data(rnaseqData, geneOverlap)

    model3Data = prepareModel3Data(geneOverlap)

    cancerTypeDrugMap = createCancerTypeDrugMap()

    meta = saveArtifacts(model2Data, model3Data, geneOverlap, cancerTypeDrugMap)

    print("\n" + "=" * 80)
    print(" " * 25 + "PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nGene overlap: {meta['num_genes']}")
    print(f"Drugs: {meta['num_drugs']}")
    print(f"\nModel 2 (Translator) samples:")
    print(f"  Train: {meta['model2_samples']['train']}")
    print(f"  Val: {meta['model2_samples']['val']}")
    print(f"  Test: {meta['model2_samples']['test']}")
    print(f"\nModel 3 (Predictor) samples:")
    print(f"  Train: {meta['model3_samples']['train']}")
    print(f"  Val: {meta['model3_samples']['val']}")
    print(f"  Test: {meta['model3_samples']['test']}")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
