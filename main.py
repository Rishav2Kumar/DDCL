'''
Created on Aug 30, 2024

Main script for the DDCL program.
'''

from configparser import ConfigParser
from os.path import exists
from os import makedirs

from constants import FOLDER_RESULTS, FOLDER_SAVED_MODEL
import plots_lib

# Access program settings from the configuration file.
config = ConfigParser()
config.read('config.ini')

program_selection = config.getint('Program', 'program')
plot_selection = config.getint('Plot_Settings', 'plot_type')

if __name__ == '__main__':

    # Check and create folders if they do not exist.
    if not exists(FOLDER_RESULTS): makedirs(FOLDER_RESULTS)
    if not exists(FOLDER_SAVED_MODEL): makedirs(FOLDER_SAVED_MODEL)

    if program_selection == 0:
        import distribution
    elif program_selection == 1:
        # Generate one of the plots.
        if plot_selection == 0:  # Loss vs Epoch
            plots_lib.diagram_combined_errorloss()
        elif plot_selection == 1:  # Batch Loss vs Step
            plots_lib.diagram_combined_errorloss(True)

        elif plot_selection == 2:  # Precision-Recall
            plots_lib.diagram_combined_precision_recall()

        elif plot_selection == 3:  # Confusion Matrix
            plots_lib.diagram_confusion_matrix()
