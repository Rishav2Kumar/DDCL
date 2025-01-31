'''
Created on Oct 2, 2024

This module defines constants and enumerations used throughout the DDCL code.
'''

from enum import IntEnum

TRAIN_STRATEGY_FULLNAMES = (
    'No Curriculum', 'Curriculum Learning - Density', 'Curriculum Learning - Point',
    'Self-Paced Learning', 'Self-Paced DDCL - Density', 'Self-Paced DDCL - Point'
)
CLASS_COL: str = 'class'
PREDICTED_COL = 'prediction'

FOLDER_DATASETS = './RealData/'
FOLDER_RESULTS = './Results/'
FOLDER_SAVED_MODEL = './SavedModel/'
DIALOG_TITLE_DATASET = 'Select a Dataset'
DIALOG_TITLE_ENSEMBLE_TRAIN = 'Select Ensemble Training Data'
DIALOG_TITLE_ENSEMBLE_TEST = 'Select Ensemble Test Data'
DIALOG_TITLE_TRAIN_METRICS = 'Select a Training Metrics results file'
DIALOG_TITLE_PREDICTIONS = 'Select a Prediction results file'

NN_TASK_TP: str = 'train_and_predict'
NN_TASK_TS: str = 'train_and_save'
NN_TASK_LP: str = 'load_and_predict'
CLS_BIN = 'binary'
CLS_MULTI = 'multi-class'


class Models(IntEnum):
    NN = 0
    SVM = 1
    RF = 2
    ENSEMBLE = 3
    ENSEMBLE_NN_SVM = 4

    @classmethod
    def get_name(cls, value):
        return Models(value).name


class TrainingStrategies(IntEnum):
    NO_CL = 0
    CL_DENSITY = 1
    CL_POINT = 2
    SELF_PACED = 3
    SP_DENSITY = 4
    SP_POINT = 5

    @classmethod
    def get_name(cls, value):
        return TrainingStrategies(value).name

    @classmethod
    def get_value(cls, name):
        return TrainingStrategies[name]
