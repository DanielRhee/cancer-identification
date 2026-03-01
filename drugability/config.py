import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from classifyModel import config as classifyConfig

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
DRUG_DATA_DIR = BASE_DIR.parent / 'drugDataset'

METHYLATION_FILE = classifyConfig.METHYLATION_FILE
CDR_FILE = classifyConfig.CDR_FILE
RANDOM_SEED = classifyConfig.RANDOM_SEED
TRAIN_RATIO = classifyConfig.TRAIN_RATIO
VAL_RATIO = classifyConfig.VAL_RATIO
TEST_RATIO = classifyConfig.TEST_RATIO

RNASEQ_FILE = DRUG_DATA_DIR / 'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv'
GDSC_EXPRESSION_FILE = DRUG_DATA_DIR / 'Cell_line_RMA_proc_basalExp.txt'
GDSC_DOSE_RESPONSE_FILE = DRUG_DATA_DIR / 'GDSC2_fitted_dose_response_27Oct23.xlsx'
CELL_LINE_DETAILS_FILE = DRUG_DATA_DIR / 'Cell_Lines_Details.xlsx'

TRANSLATOR_ENCODER_DIMS = [5000, 2048, 1024, 256]
TRANSLATOR_DECODER_DIMS = [256, 1024, 2048]
TRANSLATOR_DROPOUT = 0.3
TRANSLATOR_BATCH_SIZE = 64
TRANSLATOR_LEARNING_RATE = 1e-3
TRANSLATOR_WEIGHT_DECAY = 1e-2
TRANSLATOR_NUM_EPOCHS = 200
TRANSLATOR_WARMUP_EPOCHS = 5
TRANSLATOR_PATIENCE = 15

PREDICTOR_HIDDEN_DIMS = [1024, 512, 256]
PREDICTOR_DROPOUT = 0.4
PREDICTOR_BATCH_SIZE = 32
PREDICTOR_LEARNING_RATE = 5e-4
PREDICTOR_WEIGHT_DECAY = 1e-2
PREDICTOR_NUM_EPOCHS = 200
PREDICTOR_WARMUP_EPOCHS = 5
PREDICTOR_PATIENCE = 15

DRUG_COVERAGE_THRESHOLD = 0.5

ARTIFACT_FILES = {
    'gene_overlap': 'geneOverlap.json',
    'translator_train_methylation': 'translatorTrainMethylation.pt',
    'translator_val_methylation': 'translatorValMethylation.pt',
    'translator_test_methylation': 'translatorTestMethylation.pt',
    'translator_train_expression': 'translatorTrainExpression.pt',
    'translator_val_expression': 'translatorValExpression.pt',
    'translator_test_expression': 'translatorTestExpression.pt',
    'expression_mean': 'expressionMean.pt',
    'expression_std': 'expressionStd.pt',
    'predictor_train_expression': 'predictorTrainExpression.pt',
    'predictor_val_expression': 'predictorValExpression.pt',
    'predictor_test_expression': 'predictorTestExpression.pt',
    'predictor_train_ic50': 'predictorTrainIc50.pt',
    'predictor_val_ic50': 'predictorValIc50.pt',
    'predictor_test_ic50': 'predictorTestIc50.pt',
    'predictor_train_mask': 'predictorTrainMask.pt',
    'predictor_val_mask': 'predictorValMask.pt',
    'predictor_test_mask': 'predictorTestMask.pt',
    'drug_names': 'drugNames.json',
    'cancer_type_drug_map': 'cancerTypeDrugMap.json',
    'translator_best_model': 'translatorBestModel.pt',
    'predictor_best_model': 'predictorBestModel.pt',
    'preprocess_meta': 'preprocessMeta.json'
}

def getArtifactPath(key):
    return ARTIFACTS_DIR / ARTIFACT_FILES[key]
