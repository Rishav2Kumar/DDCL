'''
Created on Mar 3, 2022

This file contains plot functions that visualise outputs from experiments.
'''

from os.path import basename, splitext

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay

from util_lib import select_filename_dialog
from constants import TrainingStrategies, TRAIN_STRATEGY_FULLNAMES, FOLDER_RESULTS, DIALOG_TITLE_TRAIN_METRICS, DIALOG_TITLE_PREDICTIONS, CLASS_COL, PREDICTED_COL
from pandas.core import strings

INDEX_ROW = 'Row'
INDEX_EPOCH = 'Epoch'
INDEX_STEP = 'Step'


def diagram_combined_errorloss(batch_loss = False):
    """
    Plots a loss vs epoch curve from multiple CSV input files containing loss values for each training epoch 
    or a batch-level loss vs step curve from files containing loss values for each training step.
    :param batch_loss: Set to True to plot a loss vs step curve. Defaults to False for loss vs epoch curve.
    """
    filepaths = select_filename_dialog(FOLDER_RESULTS, DIALOG_TITLE_TRAIN_METRICS, True)

    if batch_loss:
        index = INDEX_STEP
        label_x = 'Training Step'
        plot_title = 'Error loss for first epoch of'
        plot_filename = 'BatchLoss_Step'
    else:
        index = INDEX_EPOCH
        label_x = 'Epoch'
        plot_title = 'Error loss for'
        plot_filename = 'Loss_Epoch'

    # Plot the error loss curves.
    loss_fig = plt.figure(figsize = (8, 6))
    ax = loss_fig.subplots()
    plt.rcParams['font.size'] = '12'

    # Create an inset axis to show 'zoomed in' content.
    inset_ax = inset_axes(ax, width = "70%", height = "40%", loc = 'center right')

    for path in filepaths:
        plot_data = pd.read_csv(path, delimiter = ',', index_col = index)
        plot_epochs = range(1, len(plot_data) + 1)

        metrics_file = splitext(basename(path))[0]
        strings = metrics_file.rsplit('-', 2)
        shortname = strings[-1]
        dataset_name = strings[1]

        try:
            scoring_method = TrainingStrategies.get_value(shortname)
        except:
            print(f'Invalid scoring method "{shortname}" in file {metrics_file}.')
        else:
            fullname = TRAIN_STRATEGY_FULLNAMES[scoring_method]

        if scoring_method in [TrainingStrategies.NO_CL, TrainingStrategies.SELF_PACED]:
            ax.plot(plot_epochs, plot_data['exp_average'], linestyle = '--', label = fullname)
            inset_ax.plot(plot_epochs[25:], plot_data['exp_average'][25:], linestyle = '--')
        else:
            ax.plot(plot_epochs, plot_data['exp_average'], label = fullname)
            inset_ax.plot(plot_epochs[25:], plot_data['exp_average'][25:])

    ax.set(xlabel = label_x, ylabel = 'Loss', title = f'{plot_title} {dataset_name.replace("_"," ")}')
    ax.legend()
    
    mark_inset(ax, inset_ax, loc1 = 2, loc2 = 4, facecolor = 'none', edgecolor = (0, 0, 0, 0.3))

    loss_fig.savefig(f'{FOLDER_RESULTS}{plot_filename}-{dataset_name}.pdf')
    plt.close()
    # plt.show()
    print('Finished plotting error loss curve.')


def diagram_combined_precision_recall():
    """
    Plots a precision-recall curve from multiple CSV input files containing actual and predicted classes for a dataset.
    """
    filepaths = select_filename_dialog(FOLDER_RESULTS, DIALOG_TITLE_PREDICTIONS, True)

    # Plot the precision recall curves.
    precision_recall_fig = plt.figure(figsize = (5.5, 5.5))
    pr_ax = precision_recall_fig.subplots()
    plt.rcParams['font.size'] = '12'
    dataset_name = ''

    for path in filepaths:
        plot_data = pd.read_csv(path, delimiter = ',', index_col = INDEX_ROW)
        class_labels = plot_data[CLASS_COL].unique()

        if len(class_labels) > 2:
            print(path)
            print('Precision Recall curve for multi-class problem is not supported.\n')
            continue
        else:
            predictions_file = splitext(basename(path))[0]
            strings = predictions_file.rsplit('-', 2)
            shortname = strings[-1]
            dataset_name = strings[1]

            try:
                scoring_method = TrainingStrategies.get_value(shortname)
            except:
                print(f'Invalid scoring method "{shortname}" in file {predictions_file}.')
            else:
                fullname = TRAIN_STRATEGY_FULLNAMES[scoring_method]

                actual = plot_data[CLASS_COL]
                predicted = plot_data[PREDICTED_COL]
                PrecisionRecallDisplay.from_predictions(actual, predicted, pos_label = 1, name = fullname, ax = pr_ax)

    if pr_ax.has_data():
        pr_ax.set(xlabel = 'Precision', ylabel = 'Recall', title = f'Precision Recall for {dataset_name.replace("_"," ")}')

        plt.savefig(f'{FOLDER_RESULTS}PrecisionRecall-{dataset_name}.pdf')
        plt.close()
        # plt.show()
        print('Finished plotting the precision-recall curve.')


def diagram_confusion_matrix():
    """
    Plots a confusion matrix using an input CSV file containing the data.
    """
    filepath = select_filename_dialog(FOLDER_RESULTS, DIALOG_TITLE_PREDICTIONS)

    if filepath:
        predictions_file = splitext(basename(filepath))[0]
        strings = predictions_file.rsplit('-', 2)
        shortname = strings[-1]
        dataset_name = strings[1]

        try:
            scoring_method = TrainingStrategies.get_value(shortname)
        except:
            print(f'Invalid scoring method "{shortname}" in file {predictions_file}.')
        else:
            fullname = TRAIN_STRATEGY_FULLNAMES[scoring_method]

            best_result_data = pd.read_csv(filepath, delimiter = ',', index_col = INDEX_ROW)
            class_labels = sorted(best_result_data[CLASS_COL].unique())
            actual = best_result_data[CLASS_COL]
            predicted = best_result_data[PREDICTED_COL]

            confusion_matrix = metrics.confusion_matrix(actual, predicted)

            confusion_matrix_fig = plt.figure(figsize = (9.4, 7.8))
            cm_ax = confusion_matrix_fig.subplots()
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_labels)
            cm_display.plot(ax = cm_ax)
            cm_display.ax_.set_title(f'Confusion Matrix for {dataset_name.replace("_", " ")} \n{fullname}')

            plt.savefig(f'{FOLDER_RESULTS}ConfusionMatrix-{dataset_name}.pdf')
            plt.close()
            # plt.show()
            print('Finished plotting the confusion matrix.')
