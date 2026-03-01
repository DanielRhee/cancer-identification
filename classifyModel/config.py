import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
DATA_DIR = BASE_DIR.parent / 'dataset'

METHYLATION_FILE = DATA_DIR / 'jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv'
CDR_FILE = DATA_DIR / 'TCGA-CDR-SupplementalTableS1.xlsx'

MANIFEST_URLS = {
    'HM450': 'https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/HM450/HM450.hg38.manifest.tsv.gz',
    'HM27': 'https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/HM27/HM27.hg38.manifest.tsv.gz'
}

NUM_PROBES = 5000
MISSINGNESS_THRESHOLD = 0.2
BETA_CLIP_MIN = 0.001
BETA_CLIP_MAX = 0.999

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 10

DROPOUT_RATE = 0.35
LABEL_SMOOTHING = 0.1

HIDDEN_DIMS = [2048, 1024, 512, 256]

ARTIFACT_FILES = {
    'train_data': 'trainData.pt',
    'val_data': 'valData.pt',
    'test_data': 'testData.pt',
    'train_labels': 'trainLabels.pt',
    'val_labels': 'valLabels.pt',
    'test_labels': 'testLabels.pt',
    'label_encoder': 'labelEncoder.json',
    'probe_names': 'probeNames.json',
    'impute_medians': 'imputeMedians.pt',
    'feature_mean': 'featureMean.pt',
    'feature_std': 'featureStd.pt',
    'class_weights': 'classWeights.pt',
    'preprocess_meta': 'preprocessMeta.json',
    'best_model': 'bestModel.pt',
    'confusion_matrix': 'confusionMatrix.png'
}

def getArtifactPath(key):
    return ARTIFACTS_DIR / ARTIFACT_FILES[key]
