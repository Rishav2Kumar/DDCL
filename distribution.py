'''
Created on Dec 17, 2021

File for DDCL code.
'''

from datetime import datetime
from configparser import ConfigParser
from os.path import basename, splitext

import pandas as pd

from util_lib import select_filename_dialog
from constants import Models, TrainingStrategies
from constants import FOLDER_DATASETS, DIALOG_TITLE_DATASET, FOLDER_RESULTS, NN_TASK_TS, TRAIN_STRATEGY_FULLNAMES, CLASS_COL, CLS_MULTI, CLS_BIN
from dataset import Dataset
import curriculum_lib as cl_lib

TIMESTAMP_FORMAT = '%Y-%m-%d_%H%M_%S'

# Read the configuration file and its access values.
config = ConfigParser()
config.read('config.ini')

model_idx = config.getint('DDCL_Settings', 'learning_model')
training_strategy = config.getint('DDCL_Settings', 'training_strategy')
number_experiments = config.getint('DDCL_Settings', 'number_experiments')
batch_size = config.getint('NN_Settings', 'batch_size')
nn_task = config.get('NN_Settings', 'nn_task')

# Use aliases for Enums.
models = Models
scoring_methods = TrainingStrategies

# Determine oversampling and ensemble options based on configuration.
oversample_quantile = True if training_strategy in [scoring_methods.CL_DENSITY, scoring_methods.CL_POINT, scoring_methods.SP_DENSITY, scoring_methods.SP_POINT] else False
use_ensemble = True if model_idx in (models.ENSEMBLE, models.ENSEMBLE_NN_SVM) else False
learning_name = models.get_name(model_idx)

# Retrieve dataset path and dataset name based on user inputs.
dataset_path = select_filename_dialog(FOLDER_DATASETS, DIALOG_TITLE_DATASET)
dataset_filename = basename(dataset_path)
dataset_name = splitext(dataset_filename)[0]

# Write experiment results summary to file.
timestamp = datetime.today()
filename = f'Metrics_{learning_name}_{dataset_name}_{timestamp.strftime(TIMESTAMP_FORMAT)}'

with open(f'{FOLDER_RESULTS}{filename}.csv', 'w') as results_file:

    if use_ensemble:
        strategies = range(len(scoring_methods)) if model_idx == models.ENSEMBLE else [training_strategy]
        text_strategy = 'Ensemble' if model_idx == models.ENSEMBLE else TRAIN_STRATEGY_FULLNAMES[training_strategy]
    else:
        strategies = [training_strategy]
        text_strategy = TRAIN_STRATEGY_FULLNAMES[training_strategy]

    print(f'Result metrics for {dataset_name} dataset.', file = results_file)
    print(f'Dataset filename: {dataset_filename}\n', file = results_file)
    print('Learning method:', learning_name, file = results_file)
    print('Training strategy:', text_strategy, file = results_file)
    if model_idx == models.NN: print('Batch size:', str(batch_size), file = results_file)
    if not use_ensemble: print('Oversampling (SMOTE):', str(oversample_quantile), file = results_file)
    dataset_name = dataset_name.replace(' ', '_')

    # Run stand-alone or ensemble experiments.
    for exp_run in range(1, number_experiments + 1):
        activate_spl = False
        model_predictions = []

        for strategy_idx in strategies:
            model_name = f'{learning_name}-{dataset_name}-{scoring_methods.get_name(strategy_idx)}'

            # Load training data from file.
            experiment_dataset = Dataset(dataset_path, CLASS_COL)

            # ----- No Curriculum -----
            if strategy_idx == scoring_methods.NO_CL:
                # Prepare data for direct training without any quantile pre-processing.
                train_dataset = pd.concat(experiment_dataset.all_data, ignore_index = True)

            # ----- Curriculum Learning - Density -----
            elif strategy_idx == scoring_methods.CL_DENSITY:
                train_dataset = experiment_dataset.prepare_curriculum(oversample_quantile, False)

            # ----- Curriculum Learning - Point -----
            elif strategy_idx == scoring_methods.CL_POINT:
                train_dataset = experiment_dataset.prepare_curriculum(oversample_quantile, True)

            # ----- Self-paced Learning -----
            elif strategy_idx == scoring_methods.SELF_PACED:
                # No data pre-processing required for self-paced learning.
                train_dataset = pd.concat(experiment_dataset.all_data, ignore_index = True)
                activate_spl = True

            # ----- Self-paced Learning + Curriculum Learning - Density -----
            elif strategy_idx == scoring_methods.SP_DENSITY:
                train_dataset = experiment_dataset.prepare_curriculum(oversample_quantile, False)

                # Enable self-paced learning during neural network training.
                activate_spl = True

            # ----- Self-paced Learning + Curriculum Learning - Point -----
            elif strategy_idx == scoring_methods.SP_POINT:
                train_dataset = experiment_dataset.prepare_curriculum(oversample_quantile, True)

                # Enable self-paced learning during neural network training.
                activate_spl = True

            # Determine values for important variables before running experiments.
            class_labels = sorted(train_dataset[CLASS_COL].unique())
            classification_type = CLS_MULTI if len(class_labels) > 2 else CLS_BIN

            if model_idx == models.NN or model_idx == models.ENSEMBLE:
                # ---------- Neural Network experiments ----------
                cl_lib.experiments_neural_network(train_dataset, CLASS_COL, batch_size, dataset_name, nn_task, model_name, exp_run, activate_spl, use_ensemble, model_predictions)
                if exp_run == number_experiments and not use_ensemble and nn_task != NN_TASK_TS:
                    cl_lib.summarise_metrics(model_name, results_file)
                    cl_lib.write_output_data(model_name)
            elif model_idx == models.SVM:
                # ---------- Support Vector Machines experiments ----------
                cl_lib.experiments_svm(train_dataset, CLASS_COL, classification_type)
                if exp_run == number_experiments:
                    cl_lib.summarise_metrics(model_name, results_file)
            elif model_idx == models.RF:
                # ---------- Random Forest experiments ----------
                cl_lib.experiments_random_forest(train_dataset, CLASS_COL, classification_type)
                if exp_run == number_experiments:
                    cl_lib.summarise_metrics(model_name, results_file)

        if model_idx == models.ENSEMBLE and nn_task != NN_TASK_TS:
            # ---------- Ensemble: 6 NNs ----------
            train_strategies = [strategy.name for strategy in scoring_methods]
            cl_lib.ensemble_classification(model_predictions, train_strategies, CLASS_COL, classification_type)
            if exp_run == number_experiments:
                cl_lib.summarise_metrics(model_name, results_file)
        elif model_idx == models.ENSEMBLE_NN_SVM:
            # ---------- Ensemble: NN + SVM ----------
            cl_lib.experiments_neural_network(train_dataset, CLASS_COL, batch_size, dataset_name, nn_task, model_name, exp_run, activate_spl, use_ensemble, model_predictions)
            cl_lib.experiments_svm(train_dataset, CLASS_COL, classification_type, use_ensemble, dataset_name, model_predictions)

            train_strategies = scoring_methods.get_name(scoring_methods.NO_CL),
            cl_lib.ensemble_classification(model_predictions, train_strategies, CLASS_COL, classification_type)
            if exp_run == number_experiments:
                cl_lib.summarise_metrics(model_name, results_file)

