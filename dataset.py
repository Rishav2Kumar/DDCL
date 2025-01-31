'''
Created on Jul 13, 2022

Dataset class
'''
import math

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


class Dataset(object):
    '''
    classdocs
    '''
    CENTROID_COL_NAME = 'dist_centroid'
    QUANTILE_COL_NAME = 'rank'
    KNN_VALUE = 3

    def __init__(self, dataset_path, target_col, string_columns = []):
        '''
        Constructor
        '''
        self.data = pd.read_csv(dataset_path, delimiter = ',')
        self.classification_col = target_col
        self.CLASS_LABELS = []
        self.max_quantiles = 0

        self.categorical_str_columns_to_int(string_columns)

        # Filter out data for each label into a List of DataFrames.
        data_grouped = []
        for label, data_group in self.data.groupby(target_col, as_index = False):
            data_grouped.append(data_group)
            self.CLASS_LABELS.append(label)

        self.all_data = data_grouped
        print('Dataset successfully loaded.')

    def categorical_str_columns_to_int(self, column_names):
        """
        Convert categorical string values to integers by replacing each value in the class' DataFrame.
        :param column_names: A List of columns names to replace values in.
        """
        for column in list(column_names):
            values = self.data[column].unique()
            values.sort()

            # Generate replacement integers for each string value.
            replacements = dict(zip(values, range(0, len(values))))
            self.data[column].replace(to_replace = replacements, inplace = True)
            self.data[column] = self.data[column].astype('int8')

    def calculate_centroids(self):
        """
        Calculate the centroids for the dataset.
        """
        print('\nCalculating centroids...')
        cluster_columns = self.data.columns.tolist()

        centroid_list = []
        for data_table in self.all_data:
            # Note: Set environment variable OMP_NUM_THREADS = 1 to avoid memory leak warning.
            kmeans_label = sklearn.cluster.KMeans(n_clusters = 1, init = 'k-means++', random_state = 0).fit(data_table[cluster_columns])

            label_centroid = pd.DataFrame(data = kmeans_label.cluster_centers_, columns = cluster_columns)
            centroid_list.append(label_centroid)

        # Calculate Euclidean distances from centroid.
        for index in range(0, len(self.all_data)):
            group_data = self.all_data[index]
            group_centroid = centroid_list[index]

            # See https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            distance_euc = numpy.linalg.norm(group_data[cluster_columns].values - group_centroid[cluster_columns].values, axis = 1)
            distance_euc = preprocessing.normalize(X = distance_euc.reshape(-1, 1), axis = 0)

            self.all_data[index] = self.all_data[index].assign(dist_centroid = distance_euc)

        print('Centroids calculation complete.')

    def determine_quantiles(self, target_quantiles = 6):
        """
        Uses a KDE plot to determine the quantiles for the dataset. 
        
        :param target_quantiles: Target limit for number of quantiles. The final number can increase due to quantile division.
        """
        fig, axs = plt.subplots(1, len(self.CLASS_LABELS))
        plt.suptitle('Distribution of dataset')
        colormap = plt.cm.tab20  # Color map for plots
        plot_colours = [colormap(i) for i in numpy.linspace(0, 1, len(self.all_data))]
        line_colours = plot_colours

        qnt_x = []
        for group_data, axis, index in zip(self.all_data, axs, range(0, len(self.all_data))):
            # Plot the distribution using the centroids.
            sns.kdeplot(data = group_data, x = self.CENTROID_COL_NAME, ax = axis, color = plot_colours[index])
            axis.set(xlabel = 'e\u0302', title = f'Class {self.CLASS_LABELS[index]}')

            plot_x_values, ymax_values = self.calculate_quantiles(axis, target_quantiles)

            # Draw the quantile lines and text for the x values of the lines.
            axis.vlines(x = plot_x_values, ymin = 0, ymax = ymax_values, colors = line_colours[index], alpha = 0.3)
            for j in range(0, len(plot_x_values)):
                y_offset = -0.25 if (j % 2 == 0) else 0.25
                axis.text(x = plot_x_values[j], y = round(ymax_values[j]) + y_offset, s = f' qx{j} = {plot_x_values[j]:.3f}', color = 'tab:orange', alpha = 0.8)

            qnt_x.append(plot_x_values)

        # plt.show(block = True) # Uncomment this line and comment plt.close() to show plot.
        plt.close()

        self.quantile_x_pts = qnt_x

    def calculate_quantiles(self, ax, target_max_quantiles):
        """
        Calculates quantiles based on the number given and returns its x values and the corresponding y max
        values for plots.
        
        :param ax: The single axes instance to use for calculating quantiles.
        :type ax:  matplotlib axes object
        :param target_max_quantiles: Targeted maximum limit for number of quantiles.
        :type target_max_quantiles: int
        """
        ax_curve = ax.lines[0]
        x_values = ax_curve.get_xdata()
        y_values = ax_curve.get_ydata()

        # Calculate the density divider lines (y values) for the quantile lines.
        y_max_rounded = math.ceil(y_values.max())
        # Divisor is the spacing to use for quantile division through horizontal lines.
        divisor = math.ceil(y_max_rounded / target_max_quantiles)
        y_cut_values = range(divisor, y_max_rounded, divisor)

        ax.hlines(y = y_cut_values, xmin = x_values.min(), xmax = x_values.max(), colors = 'r', linestyles = 'dashed', alpha = 0.4)

        #===========================================================================
        # Compute the locations where the quantile line intersects the plot's curve.
        # These intersection points (x values) are used to create the quantiles.
        #===========================================================================
        qnt_x_values = []
        for cut_point in y_cut_values:
            # 1. Calculate the difference of the quantile line and all y values
            # with the corresponding signs using numpy's sign() function.
            subtract_with_sign = numpy.sign(cut_point - y_values)

            # 2. Use diff() to determine all positions where the sign changes (i.e. the lines intersect).
            intersect_loc = numpy.diff(subtract_with_sign)

            # 3. Get the exact indices of the intersection points through argwhere()
            # and combine into single list using flatten().
            indices = numpy.argwhere(intersect_loc).flatten()

            # 4. Lookup the x value of the intersection points.
            quantiles = x_values[indices]

            # Filter out negative values if present.
            for index in range(0, len(quantiles)):
                if quantiles[index] < 0:
                    quantiles[index] = 0

            qnt_x_values.extend(quantiles)
        # Lastly, add the maximum and minimum x value to ensure that the quantiles cover all data points.
        qnt_x_values.append(x_values.max())
        qnt_x_values.append(0)

        # Remove any duplicate values.
        qnt_x_values = list(set(qnt_x_values))
        qnt_x_values = sorted(qnt_x_values)

        # Determine y values for plotting quantile lines.
        qnt_y_max_values = []
        for i in range(len(qnt_x_values)):
            qnt_y_max_values.append(numpy.interp(qnt_x_values[i], x_values, y_values))

        return qnt_x_values, qnt_y_max_values

    def assign_quantiles(self):
        print('\nAssigning data points to quantiles...')
        summary_data = {}

        for group_data, qnt, label in zip(self.all_data, self.quantile_x_pts, self.CLASS_LABELS):
            number_qnt = self.data_points_to_quantiles(group_data, qnt)

            # Generate a summary of the number of data points in each quantile.
            col_values = []
            for i in range(0, number_qnt):
                label_qnt_n_count = (group_data[self.QUANTILE_COL_NAME] == i + 1).sum()
                col_values.append(label_qnt_n_count)
                summary_data[label] = col_values

            if number_qnt > self.max_quantiles:
                self.max_quantiles = number_qnt

        # Ensure that all columns have the same number of values.
        for key in summary_data:
            if len(summary_data[key]) != self.max_quantiles:
                summary_values = summary_data[key]
                zeroes_to_add = self.max_quantiles - len(summary_values)
                # Append zeroes to summary list for quantiles that do not exist for a label.
                summary_data[key] = list(numpy.lib.pad(summary_values, (0, zeroes_to_add), 'constant'))

        # Store summary in DataFrame with the total count of each quantile.
        quantile_summary = pd.DataFrame(data = summary_data, index = range(1, self.max_quantiles + 1))
        quantile_summary = quantile_summary.assign(quantile_total = quantile_summary.sum(axis = 1))
        quantile_summary.index.name = 'Quantile'
        print('Quantile assignment complete. Summary of points:\n', quantile_summary)
        print('Sum of quantile_total column:', quantile_summary.quantile_total.sum())
        print('Total rows in dataset:', len(self.data.index), '\n')

        # Oversampled quantiles step 1: Determine the quantiles with the lowest number of points. Include only those with sufficient number of points for oversampling.
        nonzero_quantiles = quantile_summary.loc[(quantile_summary > self.KNN_VALUE).any(axis = 1)]
        num_small_quantiles = int(len(nonzero_quantiles) * 0.5)  # Take half of the total number of quantiles.
        self.smallest_quantiles = nonzero_quantiles['quantile_total'].nsmallest(num_small_quantiles).index.to_list()
        # print(self.smallest_quantiles)

    def data_points_to_quantiles(self, points_df, quantile_list):
        """
        Ranks centroid distances into quantiles based on density.
        
        :param points_df: Euclidean distances from centroids for one class.
        :type points_df: DataFrame
        :param quantile_list: Quantile x values for one class.
        :type quantile_list: List
        """
        # Rank the data points into quantiles.
        quantile_counter = 0
        points_remaining = quantile_list.copy()

        for index in range(0, len(quantile_list) - 1, 1):
            start_pt = quantile_list[index]

            # Retrieve all points excluding the starting point.
            points_remaining.remove(start_pt)

            # Quantile end point which is next minimum value in the points List.
            end_pt = min(points_remaining)

            # Assign data points to quantile.
            quantile_counter += 1
            points_df.loc[points_df[self.CENTROID_COL_NAME].between(start_pt, end_pt), self.QUANTILE_COL_NAME] = quantile_counter

        return quantile_counter

    def divide_by_quantile(self, oversample = False, point_scoring = False):
        # Combine data points for all quantiles across each label into individual DataFrames by density (1, 2, etc.).
        data_by_density = []

        for j in range(0, self.max_quantiles):
            quantile = j + 1
            overall_data_qnt_n = []
            skip_counter = 0
            for group_data in self.all_data:
                # Extract rows for quantile n (1, 2, 3, etc.) from each label into list.
                rows_qnt_n = group_data.loc[group_data[self.QUANTILE_COL_NAME] == quantile]
                overall_data_qnt_n.append(rows_qnt_n)
                if len(rows_qnt_n) == 0:
                    skip_counter += 1

            if skip_counter == len(self.CLASS_LABELS):
                continue

            # Create DataFrame containing all data points for a specific quantile.
            quantile_n_df = pd.concat(overall_data_qnt_n, ignore_index = True)
            quantile_n_df[self.classification_col] = quantile_n_df[self.classification_col].astype('category')
            quantile_n_df[self.classification_col] = quantile_n_df[self.classification_col].cat.set_categories(self.CLASS_LABELS)

            num_instances = quantile_n_df[self.classification_col].value_counts()
            sufficient_samples = all(count > self.KNN_VALUE for count in num_instances)

            # Oversampled quantiles step 2: Apply SMOTE to the smallest quantiles provided they have sufficient number of samples.
            if oversample and (quantile in self.smallest_quantiles) and sufficient_samples:
                oversample_smote = SMOTE(sampling_strategy = 'minority', k_neighbors = self.KNN_VALUE)
                data_resampled, label_resampled = oversample_smote.fit_resample(quantile_n_df.drop(self.classification_col, axis = 'columns'), quantile_n_df[self.classification_col])
                print(f'Data for Quantile {quantile} oversampled from {len(quantile_n_df)} points to {len(data_resampled)} using SMOTE.')
                quantile_n_df = pd.concat([data_resampled, label_resampled], axis = 1)

            data_by_density.append(quantile_n_df)

        # Combine individual densities into a single DataFrame for training.
        if point_scoring:
            combined_data = pd.concat(data_by_density).sort_index().reset_index(drop = True)
        else:
            # Sort the densities from highest to lowest.
            data_by_density = sorted(data_by_density, key = lambda x:len(x), reverse = True)

            combined_data = pd.concat(data_by_density, ignore_index = True)

        combined_data.drop(columns = [self.CENTROID_COL_NAME, self.QUANTILE_COL_NAME], inplace = True)

        return combined_data

    def prepare_curriculum(self, oversample_quantile, score_data_points):
        """
        Prepare the dataset using curriculum learning.
        :param oversample_quantile: Auto-assigned value that determines if oversampling is applied.
        :param score_data_points: Set to False for 'Density' scoring and True for 'Point' scoring.
        """
        # Calculate centroids.
        self.calculate_centroids()

        # Plot the distribution and determine quantiles.
        self.determine_quantiles()

        # Assign points to quantiles.
        self.assign_quantiles()

        # Divide data points according to their quantiles.
        all_data_by_density = self.divide_by_quantile(oversample_quantile, score_data_points)

        return all_data_by_density
