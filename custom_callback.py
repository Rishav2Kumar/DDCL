'''
Created on Feb 21, 2023

Custom_Callback class
'''

from tensorflow import keras

import numpy
import pandas as pd

LAMBDA_MAX = 0.6


class Custom_Callback(keras.callbacks.Callback):
    '''
    classdocs
    '''

    def __init__(self, batch_loss, spl_active = False, dataset_array = None, dataset_dataframe = None, batch_size = 32):
        '''
        Constructor
        '''
        super(Custom_Callback, self).__init__()
        self.step_losses = []
        self.batchloss = batch_loss

        # SPL variables
        self.self_paced_learning = spl_active
        if (self.self_paced_learning):
            self.training_data = dataset_array
            self.train_dataframe = dataset_dataframe
            self.spl_batchsize = batch_size
            self.spl_batch_order = []
            self.spl_threshold = 0.4
            self.spl_growing_factor = 1.10
            self.spl_apply_threshold = False
            self.spl_using_subset = False
            self.original_data = dataset_array.copy()
            self.original_dataframe = dataset_dataframe.copy()

    def on_train_batch_end(self, batch, logs = None):
        self.step_losses.append(logs['loss'])

        if (self.self_paced_learning):
            if logs['loss'] < self.spl_threshold and (self.spl_apply_threshold and self.spl_using_subset == False):
                # Use data batches considered 'Easy'.
                # print(f'Batch {batch} is Easy. Loss value =', logs['loss'])
                self.spl_batch_order.append(batch)
            else:
                # 'Hard' batches are skipped until their difficulty decreases.
                # print(f'Batch {batch} is Hard. Loss value =', logs['loss'])
                pass

    def on_epoch_begin(self, epoch, logs = None):
        if (self.self_paced_learning):
            if self.spl_apply_threshold == False and epoch > 9:
                self.spl_apply_threshold = True

    def on_epoch_end(self, epoch, logs = None):
        # Record the batch level losses for only the first epoch.
        if epoch == 0:
            self.batchloss.extend(self.step_losses)

        if (self.self_paced_learning):
            # Reset training data to default and include all batches.
            if (logs['loss'] < self.spl_threshold) and self.spl_using_subset == True:
                self.training_data.clear()
                self.training_data.extend(self.original_data)
                self.train_dataframe = self.original_dataframe
                self.train_dataframe.reset_index(drop = True, inplace = True)
                # print(f'Training reset data to default: Loss = {logs["loss"]}, Threshold = {self.spl_threshold}')
                self.spl_using_subset = False
                self.spl_batch_order.clear()

            if len(self.spl_batch_order) != 0:
                # Determine the new batch order based on the losses.
                new_batch_order = [batch_num for _, batch_num in sorted(zip(self.step_losses, self.spl_batch_order))]

                if self.spl_threshold < LAMBDA_MAX:
                    self.spl_threshold = self.spl_threshold * self.spl_growing_factor

                num_of_samples = len(self.train_dataframe)
                batch_indexes = [b_idx for b_idx in range(0, num_of_samples, self.spl_batchsize)]
                batch_indexes.append(num_of_samples)  # The end slice index is exclusive, hence add the total samples.

                # Slice original data into multiple dataframes where each one represents a single batch.
                dataframe_list = [self.train_dataframe.iloc[batch_indexes[s_idx]:batch_indexes[s_idx + 1]] for s_idx in range(len(batch_indexes) - 1)]

                # Reorder the list of dataframes according to new batch order.
                reordered_data = [dataframe_list[batchnum] for batchnum in new_batch_order]
                new_dataframe = pd.concat(objs = [*reordered_data], axis = 0)

                num_total_cols = len(new_dataframe.columns)
                num_data_cols = num_total_cols - 1
                target_col = new_dataframe.columns[num_total_cols - 1]

                # Convert the filtered Dataframe containing 'Easy' samples to Numpy array.
                features_array = new_dataframe.drop(target_col, axis = 'columns').to_numpy()
                features_array = numpy.split(features_array, num_data_cols, axis = 1)
                for i in range(len(features_array)):
                    concat_column = numpy.concatenate(features_array[i])
                    features_array[i] = concat_column
                labels_array = new_dataframe[target_col].to_numpy()
                new_training_data = (features_array, labels_array)

                #======================================================
                # Overwrite current training data with SPL defined data
                #======================================================
                self.training_data.clear()
                self.training_data.extend(new_training_data)
                self.train_dataframe = new_dataframe
                self.train_dataframe.reset_index(drop = True, inplace = True)
                self.spl_using_subset = True

            self.spl_batch_order.clear()

        # Clear values for step_losses at the end of each epoch.
        self.step_losses.clear()

