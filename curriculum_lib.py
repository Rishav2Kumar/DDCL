'''
Created on Mar 28, 2022

This file contains functions for training a dataset and performing experiments.
'''

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.losses import SparseCategoricalCrossentropy
from keras.losses import BinaryCrossentropy
from keras.layers import Normalization

import numpy
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from constants import FOLDER_SAVED_MODEL, CLS_MULTI, CLS_BIN, PREDICTED_COL, FOLDER_RESULTS, FOLDER_DATASETS, NN_TASK_TP, NN_TASK_TS, NN_TASK_LP
from custom_callback import Custom_Callback

TIMESTAMP_FORMAT = '%Y-%m-%d_%H%M_%S'

hyperparameter_values = [{'Dataset': 'Breast_Cancer_Diagnostic', 'Dropout': 0.4, 'num_layers': 1, 'Layer_0': 400, 'Activation_0': 'tanh'}
                         , {'Dataset': 'Cancer', 'Dropout': 0.0, 'num_layers': 5, 'Layer_0': 16, 'Activation_0': 'relu',
                            'Layer_1': 336, 'Activation_1': 'relu', 'Layer_2': 16, 'Activation_2': 'relu', 'Layer_3': 16, 'Activation_3': 'relu', 'Layer_4': 16, 'Activation_4': 'relu'}
                         , {'Dataset': 'Haberman', 'Dropout': 0.1, 'num_layers': 5, 'Layer_0': 352, 'Activation_0': 'relu',
                            'Layer_1': 512, 'Activation_1': 'tanh', 'Layer_2': 16, 'Activation_2': 'relu', 'Layer_3': 16, 'Activation_3': 'relu', 'Layer_4': 16, 'Activation_4': 'relu'}
                         , {'Dataset': 'Liver_Disorder', 'Dropout': 0.2, 'num_layers': 5, 'Layer_0': 112, 'Activation_0': 'relu',
                            'Layer_1': 224, 'Activation_1': 'tanh', 'Layer_2': 208, 'Activation_2': 'relu', 'Layer_3': 16, 'Activation_3': 'relu', 'Layer_4': 16, 'Activation_4': 'relu'}
                         , {'Dataset': 'Pima_Indians_Diabetes', 'Dropout': 0.4, 'num_layers': 5, 'Layer_0': 16, 'Activation_0': 'relu',
                            'Layer_1': 16, 'Activation_1': 'relu', 'Layer_2': 16, 'Activation_2': 'relu', 'Layer_3': 16, 'Activation_3': 'relu', 'Layer_4': 16, 'Activation_4': 'relu'}
                         , {'Dataset': 'New-Thyroid', 'Dropout': 0.5, 'num_layers': 5, 'Layer_0': 16, 'Activation_0': 'relu',
                            'Layer_1': 16, 'Activation_1': 'relu', 'Layer_2': 16, 'Activation_2': 'relu', 'Layer_3': 16, 'Activation_3': 'relu', 'Layer_4': 16, 'Activation_4': 'relu'}
                         , {'Dataset': 'ISOLET', 'Dropout': 0.0, 'num_layers': 1, 'Layer_0': 512, 'Activation_0': 'relu'}
                         , {'Dataset': 'Optical_Recognition_Handwritten_Digits', 'Dropout': 0.3, 'num_layers': 5, 'Layer_0': 512, 'Activation_0': 'tanh',
                            'Layer_1': 192, 'Activation_1': 'relu', 'Layer_2': 160, 'Activation_2': 'tanh', 'Layer_3': 480, 'Activation_3': 'relu', 'Layer_4': 80, 'Activation_4': 'tanh'}
                         , {'Dataset': 'Clean1', 'Dropout': 0.1, 'num_layers': 5, 'Layer_0': 384, 'Activation_0': 'relu',
                            'Layer_1': 160, 'Activation_1': 'tanh', 'Layer_2': 240, 'Activation_2': 'tanh', 'Layer_3': 16, 'Activation_3': 'relu', 'Layer_4': 16, 'Activation_4': 'relu'}
                         , {'Dataset': 'Banknote', 'Dropout': 0.4, 'num_layers': 1, 'Layer_0': 480, 'Activation_0': 'tanh'}]

metrics_list = []
best_run_accuracy = 0
best_run_prediction_data = None
batchlosses_dict = {}
trainmetrics_dict = {}


def dataframe_to_tf_dataset(input_dataframe, CLASS_COL):
    """
    Converts a pandas DataFrame into a TensorFlow Dataset suitable for model training,
    separating the specified class label column from the features.

    Args:
        input_dataframe (pd.DataFrame): The input DataFrame containing features and labels.
        CLASS_COL (str): The name of the column containing class labels.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset of (features, labels) tuples.
    """
    dataframe = input_dataframe.copy()
    class_labels = dataframe.pop(CLASS_COL)

    # Create tf Dataset from Dataframe rows and classification labels
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), class_labels))

    return dataset


def dataset_nn_prepare(training_df, training_dataset):
    """
    Prepares Keras model input layers and encoded feature tensors from a training DataFrame and dataset.
    Args:
        training_df (pd.DataFrame): The DataFrame containing the training data, where each column represents a feature.
        training_dataset (tf.data.Dataset): The TensorFlow dataset corresponding to the training data, used for feature encoding.
    Returns:
        tuple: A tuple containing:
            - all_inputs (list): List of Keras Input layers, one for each feature column in the DataFrame.
            - all_features (tf.Tensor): Concatenated tensor of all encoded features, suitable as model input.
    Notes:
        - This function assumes the existence of an `encode_numerical_feature` function for feature encoding.
        - All features are concatenated into a single tensor named 'All_Inputs'.
    """
    all_inputs = []
    # Prepare model inputs from dataset.
    for column_name in training_df.columns:
        data_type = training_df[column_name].dtype
        model_input = keras.Input(shape=(1,), name = column_name, dtype = data_type)
        all_inputs.append(model_input)

    encoded_features = []
    # Encode training features from inputs.
    for input_feature in all_inputs:
        model_feature = encode_numerical_feature(input_feature, input_feature.name, training_dataset)
        encoded_features.append(model_feature)

    all_features = layers.concatenate(encoded_features, name = 'All_Inputs')

    return all_inputs, all_features


def encode_numerical_feature(feature, col_name, dataset):
    """
    Encodes a numerical feature using normalization.
    This function creates a Keras Normalization layer, adapts it to the distribution of the specified column in the provided dataset, and applies the normalization to the given feature tensor.
    Args:
        feature (tf.Tensor): The input tensor representing the feature to be normalized.
        col_name (str): The name of the column in the dataset to be normalized.
        dataset (tf.data.Dataset): The dataset containing the feature column, where each element is a tuple (features, label).
    Returns:
        tf.Tensor: The normalized feature tensor.
    """
    
    normalizer = Normalization(axis = None, name = (col_name + '_Normalised'))

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[col_name])

    normalizer.adapt(feature_ds)

    encoded_feature = normalizer(feature)
    return encoded_feature


def dataset_create_model(model_inputs, model_features, output_dimension, tuned_hyperparameters):
    """
    Creates and compiles a Keras model based on the provided input layer, feature layer, output dimension, 
    and a dictionary of tuned hyperparameters.
    The function dynamically builds a feedforward neural network with a variable number of hidden layers, 
    units, and activations as specified in `tuned_hyperparameters`. The output layer's activation and loss 
    function are chosen based on the `output_dimension` (softmax/SparseCategoricalCrossentropy for 
    multi-class, sigmoid/BinaryCrossentropy for binary).
    Args:
        model_inputs (tf.keras.layers.Layer): The input layer for the model.
        model_features (tf.keras.layers.Layer): The feature extraction or preprocessing layer.
        output_dimension (int): The number of output units/classes.
        tuned_hyperparameters (dict): Dictionary containing hyperparameters such as:
            - 'num_layers': Number of hidden layers.
            - 'Layer_{i}': Number of units in the i-th hidden layer.
            - 'Activation_{i}': Activation function for the i-th hidden layer.
            - 'Dropout': Dropout rate to apply after the last hidden layer.
    Returns:
        tf.keras.Model: The compiled Keras model ready for training.
    """
    # Change model parameters depending on the output dimension.
    if output_dimension >= 3:
        activation_output = 'softmax'
        model_loss = SparseCategoricalCrossentropy()
    else:
        activation_output = 'sigmoid'
        model_loss = BinaryCrossentropy()

    # Load tuned hyperparameters for model.
    previous_layer = model_features
    for i in range(tuned_hyperparameters['num_layers']):
        new_layer = layers.Dense(units = tuned_hyperparameters[f'Layer_{i}'], activation = tuned_hyperparameters[f'Activation_{i}'], name = f'Hidden_{i+1}')(previous_layer)
        previous_layer = new_layer

    layer_dropout = layers.Dropout(tuned_hyperparameters['Dropout'], name = 'Dropout')(previous_layer)
    layer_output = layers.Dense(units = output_dimension, activation = activation_output, name = 'Output')(layer_dropout)

    model = keras.Model(inputs = model_inputs, outputs = layer_output)
    model.compile(optimizer = 'adam', loss = model_loss, metrics = ['accuracy'])

    return model


def dataset_nn_train(model, training_dataset, validation_dataset, num_epochs, spl_active, batchSize = None, training_df = None, saved_name = None):
    """
    Trains a neural network model on the provided dataset with optional self-paced learning and custom callbacks.
    Args:
        model (tf.keras.Model): The neural network model to be trained.
        training_dataset (tuple or tf.data.Dataset): Training data. If `spl_active` is True, should be a tuple (inputs, targets); otherwise, a dataset.
        validation_dataset (tuple or tf.data.Dataset): Validation data for evaluating the model during training.
        num_epochs (int): Number of epochs to train the model.
        spl_active (bool): Flag to activate self-paced learning (SPL) with custom callback behavior.
        batchSize (int, optional): Batch size for training. Required if `spl_active` is True.
        training_df (pd.DataFrame, optional): DataFrame containing training data, used by the custom callback if SPL is active.
        saved_name (str, optional): If provided, saves the trained model with this name.
    Returns:
        model (tf.keras.Model): The trained model.
        batch_loss (list): List of batch losses collected during training.
        train_metrics (list): List of training loss values for each epoch.
    Side Effects:
        - Prints training progress and model save confirmation.
        - Saves the trained model to disk if `saved_name` is provided.
    """
    batch_loss = []
    custom_callback = Custom_Callback(batch_loss, spl_active, training_dataset, training_df, batchSize)

    print('Starting training ...')
    if spl_active:
        train_record = model.fit(training_dataset[0], training_dataset[1], batch_size = batchSize, epochs = num_epochs, validation_data = validation_dataset, verbose = 2, callbacks = [custom_callback])
    else:
        train_record = model.fit(training_dataset, epochs = num_epochs, validation_data = validation_dataset, verbose = 2, callbacks = [custom_callback])

    train_metrics = train_record.history['loss']

    if saved_name != None:
        model.save(f'{FOLDER_SAVED_MODEL}{saved_name}.keras')
        print(f'\nSaved trained model to file: {saved_name}.keras')

    return model, batch_loss, train_metrics


def dataset_nn_predict(model, test_data, class_label_col, CLASS_LABELS):
    """
    Performs prediction on a test dataset using a neural network model and computes classification metrics.
    Args:
        model: Trained neural network model with a `predict` method.
        test_data (pd.DataFrame): DataFrame containing the test data, including the class label column.
        class_label_col (str): Name of the column in `test_data` containing the true class labels.
        CLASS_LABELS (list): List of possible class labels.
    Returns:
        tuple:
            - metrics (dict): Dictionary containing classification metrics computed from the predictions.
            - prediction_data (pd.DataFrame): DataFrame containing the actual labels, predicted labels, and (for multiclass) prediction probabilities for each class.
    Notes:
        - For multiclass classification, the function outputs probabilities for each class and the predicted label.
        - For binary classification, the function outputs the predicted probability and the rounded prediction.
        - The function modifies `test_data` in place by dropping the class label column.
    """
    actual_labels = test_data[class_label_col]
    test_data.drop(columns = [class_label_col], inplace = True)

    test_input = {}
    for column_name in test_data.columns:
        test_input[column_name] = tf.convert_to_tensor(test_data[column_name])

    # Perform classification on new data.
    predictions = model.predict(test_input)
    results_data = [confidence for confidence in predictions]

    if len(CLASS_LABELS) > 2:
        classification_type = CLS_MULTI
        PREDICTION_COL_NAMES = [('label_' + str(label)) for label in CLASS_LABELS]
        output_df = pd.DataFrame(data = results_data, columns = PREDICTION_COL_NAMES)

        # Add the actual and predicted labels to the output. Then, determine which label the model predicted.
        output_df = pd.concat(objs = [actual_labels, output_df], axis = 1)
        output_df = output_df.assign(prediction = output_df[PREDICTION_COL_NAMES].idxmax(axis = 'columns'))
        rename_dict = dict(zip(PREDICTION_COL_NAMES, CLASS_LABELS))
        output_df[PREDICTED_COL].replace(to_replace = rename_dict, inplace = True)
    else:
        classification_type = CLS_BIN
        PREDICTION_COL_NAMES = ['probability']
        output_df = pd.DataFrame(data = results_data, columns = PREDICTION_COL_NAMES)

        # Add the actual labels and predicted probabilities to the output.
        output_df = pd.concat(objs = [actual_labels, output_df], axis = 1)
        output_df = output_df.assign(prediction = output_df[PREDICTION_COL_NAMES].round(0))

    prediction_data = output_df.drop(columns = PREDICTION_COL_NAMES)

    # Compute classification metrics using results.
    metrics = compute_result_metrics(output_df, class_label_col, PREDICTED_COL, classification_type)

    return metrics, prediction_data


def compute_result_metrics(results_data, actual, predicted, classification_type):
    """
    Compute the Accuracy, Recall, Precision and F1 score for the given data.
    :param results_data: A DataFrame containing the results data.
    :param actual: Name of the column for actual labels.
    :param predicted: Name of the column for predicted labels.
    :param classification_type: Type of classification problem based on actual labels.
    """
    averaging_type = 'weighted' if classification_type == CLS_MULTI else CLS_BIN

    metric_accuracy = metrics.accuracy_score(results_data[actual], results_data[predicted])
    metric_recall = metrics.recall_score(results_data[actual], results_data[predicted], average = averaging_type)
    metric_precision = metrics.precision_score(results_data[actual], results_data[predicted], average = averaging_type)
    metric_f1_score = metrics.f1_score(results_data[actual], results_data[predicted], average = averaging_type)

    classification_metrics = {'Accuracy': metric_accuracy, 'Recall': metric_recall, 'Precision': metric_precision, 'F1_Score': metric_f1_score}
    print('\n-------------------- Classification Results --------------------')
    print(f'Accuracy {metric_accuracy}\tRecall {metric_recall}\tPrecision {metric_precision}\tF1_Score {metric_f1_score}', '\n')

    return classification_metrics


def summarise_metrics(model_name, results_file):
    # Summarise metrics after all experiment runs are completed.
    summary_dataframe = pd.DataFrame(data = metrics_list, index = range(1, len(metrics_list) + 1))
    summary_dataframe.index.name = 'Experiment'
    average = summary_dataframe['Accuracy'].mean()
    worst = summary_dataframe['Accuracy'].min()
    best = summary_dataframe['Accuracy'].max()
    std_dev = summary_dataframe['Accuracy'].std(ddof = 0)  # Delta Degrees of Freedom set to 0 for population standard deviation.

    if results_file != None:
        print('\n', summary_dataframe, file = results_file)
        print('\nWorst Accuracy \t Best Accuracy \t Average Accuracy \t Std Dev (Population)', file = results_file)
        print(worst, '\t', best, '\t', average, '\t', std_dev, file = results_file)

    # Save prediction data for Precision-Recall curve to file.
    best_run_prediction_data.index.name = 'Row'
    best_run_prediction_data.to_csv(f'{FOLDER_RESULTS}Predict-{model_name}.csv', encoding = 'utf-8', index = True)


def write_output_data(model_name):
    batchloss_df = pd.DataFrame.from_dict(batchlosses_dict)
    batchloss_df.index.name = 'Step'
    batchloss_df = batchloss_df.assign(exp_average = batchloss_df.mean(axis = 1))
    batchloss_df.to_csv(f'{FOLDER_RESULTS}BatchLoss-{model_name}.csv', encoding = 'utf-8', index = True)
    print('Saved batch loss data to file.')

    trainmetrics_df = pd.DataFrame.from_dict(trainmetrics_dict)
    trainmetrics_df.index.name = 'Epoch'
    trainmetrics_df = trainmetrics_df.assign(exp_average = trainmetrics_df.mean(axis = 1))
    trainmetrics_df.to_csv(f'{FOLDER_RESULTS}TrainMetric-{model_name}.csv', encoding = 'utf-8', index = True)
    print('Saved epoch loss data to file.')


def experiments_neural_network(train_dataset, CLASS_COL, batch_size, dataset_name, task, model_name, exp_run, activate_spl, ensemble_mode, predictions_list):
    #==============================================
    # Neural Network training and/or classification
    #==============================================
    class_labels = sorted(train_dataset[CLASS_COL].unique())
    # Set output_dimension to 1 for binary classification or to the unique number of labels for multiclass classification.
    output_dimension = 1 if len(class_labels) == 2 else len(class_labels)

    validation_df = train_dataset.sample(frac = 0.2)
    training_df = train_dataset.drop(validation_df.index)
    # ---------- Test data selection for classification experiments ----------
    if ensemble_mode:
        test_data = pd.read_csv(f'{FOLDER_DATASETS}ensemble/{dataset_name}_test.csv', delimiter = ',')
    else:
        test_data = training_df.sample(frac = 0.1)
        training_df = training_df.drop(test_data.index)
        # Reset index for use in testing.
        test_data.reset_index(drop = True, inplace = True)

    training_size = len(training_df)
    validation_size = len(validation_df)

    print('\n---------- Tensorflow model training and/or classification ----------')
    print('Using', training_size, 'samples for training,', validation_size, 'for validation and', len(test_data), 'for testing \n')

    training_dataset = dataframe_to_tf_dataset(training_df, CLASS_COL)
    validation_dataset = dataframe_to_tf_dataset(validation_df, CLASS_COL)

    training_dataset = training_dataset.batch(batch_size, drop_remainder = True)
    validation_dataset = validation_dataset.batch(batch_size, drop_remainder = True)

    nn_inputs, nn_features = dataset_nn_prepare(training_df.drop(CLASS_COL, axis = 'columns'), training_dataset)

    # Convert training data to Numpy for SPL.
    if activate_spl:
        training_dataset = training_df
        training_df.reset_index(drop = True, inplace = True)

        features_array = training_dataset.drop(CLASS_COL, axis = 'columns').to_numpy()
        num_columns = len(training_dataset.columns) - 1
        features_array = numpy.split(features_array, num_columns, axis = 1)
        for i in range(len(features_array)):
            concat_column = numpy.concatenate(features_array[i])
            features_array[i] = concat_column
        labels_array = training_dataset[CLASS_COL].to_numpy()
        training_dataset = [features_array, labels_array]

    if task == NN_TASK_TP:
        dataset_hyperparameters = next(parameters for parameters in hyperparameter_values if (parameters['Dataset'] == dataset_name))

        # ----- Build model and train for experiment run -----
        nn_model = dataset_create_model(nn_inputs, nn_features, output_dimension, dataset_hyperparameters)
        trained_model, loss_list, training_metrics = dataset_nn_train(nn_model, training_dataset, validation_dataset, 200, activate_spl, batch_size, training_df)

        # Log training data for this experiment run.
        batchlosses_dict[f'Exp{exp_run}'] = loss_list
        trainmetrics_dict[f'Exp{exp_run}'] = training_metrics

        # Predict on test data using trained model.
        summary_metrics, predictions = dataset_nn_predict(trained_model, test_data, CLASS_COL, class_labels)
        metrics_list.append(summary_metrics)

        global best_run_accuracy, best_run_prediction_data
        # Save data from the best result for plots.
        if summary_metrics['Accuracy'] > best_run_accuracy:
            best_run_accuracy = summary_metrics['Accuracy']
            best_run_prediction_data = predictions

    elif task == NN_TASK_TS:
        dataset_hyperparameters = next(parameters for parameters in hyperparameter_values if (parameters['Dataset'] == dataset_name))
        # ----- Train model and save weights -----
        weight_save_name = 'Model_' + model_name
        nn_model = dataset_create_model(nn_inputs, nn_features, output_dimension, dataset_hyperparameters)
        trained_model, loss_list, training_metrics = dataset_nn_train(nn_model, training_dataset, validation_dataset, 600, activate_spl, batch_size, training_df, weight_save_name)

        # Log training data for this experiment run.
        batchlosses_dict[f'Exp{exp_run}'] = loss_list
        trainmetrics_dict[f'Exp{exp_run}'] = training_metrics

    elif task == NN_TASK_LP:
        dataset_hyperparameters = next(parameters for parameters in hyperparameter_values if (parameters['Dataset'] == dataset_name))
        # Create model instance and load saved weights.
        saved_model = dataset_create_model(nn_inputs, nn_features, output_dimension, dataset_hyperparameters)
        saved_model.load_weights(f'{FOLDER_SAVED_MODEL}Model_{model_name}.keras')

        summary_metrics, predictions = dataset_nn_predict(saved_model, test_data, CLASS_COL, class_labels)

        # Store prediction data for model.
        predictions_list.append(predictions)


def experiments_svm(train_dataset, CLASS_COL, classification_type, use_ensemble = False, dataset_name = '', predictions_list = None):

    # Normalise data using sklearn MinMaxScaler.
    min_max_scaler = preprocessing.MinMaxScaler()
    normalised_values = min_max_scaler.fit_transform(train_dataset.values)
    normalised_training_data = pd.DataFrame(data = normalised_values, columns = train_dataset.columns.tolist())
    normalised_training_data[CLASS_COL] = normalised_training_data[CLASS_COL].astype('int8')

    features = normalised_training_data.drop(CLASS_COL, axis = 'columns')

    if use_ensemble:
        # Prepare the isolated test data for ensemble learners.
        test_dataset = pd.read_csv(f'{FOLDER_DATASETS}ensemble/{dataset_name}_test.csv', delimiter = ',')
        normalised_values = min_max_scaler.fit_transform(test_dataset.values)
        normalised_testing_data = pd.DataFrame(data = normalised_values, columns = test_dataset.columns.tolist())
        normalised_testing_data[CLASS_COL] = normalised_testing_data[CLASS_COL].astype('int8')

        features_train = features
        features_test = normalised_testing_data.drop(CLASS_COL, axis = 'columns')
        class_labels_train = normalised_training_data[CLASS_COL]
        class_labels_test = normalised_testing_data[CLASS_COL]
    else:
        # Split dataset into training and testing sets.
        features_train, features_test, class_labels_train, class_labels_test = train_test_split(features, normalised_training_data[CLASS_COL], test_size = 0.3)

    if classification_type == CLS_BIN:
        svm_model = svm.SVC(cache_size = 1000)
    elif classification_type == CLS_MULTI:
        svm_model = svm.SVC(cache_size = 1000, decision_function_shape = 'ovo')

    # ----- Train the model for experiments -----
    svm_model.fit(features_train, class_labels_train)

    # Predict on testing data.
    predicted_class_labels = svm_model.predict(features_test)

    class_labels_test.reset_index(drop = True, inplace = True)
    # Create output DataFrame from results.
    output_df = pd.DataFrame(data = predicted_class_labels, columns = [PREDICTED_COL])
    output_df = pd.concat(objs = [class_labels_test, output_df], axis = 1)

    if use_ensemble:
        # Store prediction data for model.
        predictions_list.append(output_df)
    else:
        summary_metrics = compute_result_metrics(output_df, CLASS_COL, PREDICTED_COL, classification_type)
        metrics_list.append(summary_metrics)

        global best_run_accuracy, best_run_prediction_data
        # Save data from the best result for plots.
        if summary_metrics['Accuracy'] > best_run_accuracy:
            best_run_accuracy = summary_metrics['Accuracy']
            best_run_prediction_data = output_df


def experiments_random_forest(train_dataset, CLASS_COL, classification_type):

    # Normalise data using sklearn MinMaxScaler.
    min_max_scaler = preprocessing.MinMaxScaler()
    normalised_values = min_max_scaler.fit_transform(train_dataset.values)
    normalised_training_data = pd.DataFrame(data = normalised_values, columns = train_dataset.columns.tolist())
    normalised_training_data[CLASS_COL] = normalised_training_data[CLASS_COL].astype('int8')

    features = normalised_training_data.drop(CLASS_COL, axis = 'columns')
    # Split dataset into training and testing sets.
    features_train, features_test, class_labels_train, class_labels_test = train_test_split(features, normalised_training_data[CLASS_COL], test_size = 0.3)

    # Create the Random Forest classifier.
    random_forest_model = RandomForestClassifier(n_estimators = 100)

    # ----- Train the model for experiments -----
    random_forest_model.fit(features_train, class_labels_train)

    # Predict on testing data.
    predicted_class_labels = random_forest_model.predict(features_test)

    class_labels_test.reset_index(drop = True, inplace = True)
    # Create output DataFrame from results.
    output_df = pd.DataFrame(data = predicted_class_labels, columns = [PREDICTED_COL])
    output_df = pd.concat(objs = [class_labels_test, output_df], axis = 1)

    summary_metrics = compute_result_metrics(output_df, CLASS_COL, PREDICTED_COL, classification_type)
    metrics_list.append(summary_metrics)

    global best_run_accuracy, best_run_prediction_data
    # Save data from the best result for plots.
    if summary_metrics['Accuracy'] > best_run_accuracy:
        best_run_accuracy = summary_metrics['Accuracy']
        best_run_prediction_data = output_df


def ensemble_classification(predictions_list, train_strategies, CLASS_COL, classification_type):
    actual_labels = []
    model_predictions = []
    ACTUAL_COL_NAME = 'actual_labels'
    PREDICT_COL_NAME = 'ensemble_prediction'

    for i in range(len(predictions_list)):
        if len(train_strategies) == 1:
            strategy = train_strategies[0]
        else:
            strategy = train_strategies[i]
        predictions = predictions_list[i]
        # Read actual class labels from file.
        if len(model_predictions) == 0:
            actual_labels = predictions[CLASS_COL]

        new_column = {strategy: predictions[PREDICTED_COL].round(0)}
        predictions = predictions.assign(**new_column)
        model_predictions.append(predictions[strategy])

    # Build ensemble dataframe with predictions from each model.
    ensemble_df = pd.concat(objs = model_predictions, axis = 1)
    mode_of_predictions = ensemble_df.mode(axis = 'columns')
    ensemble_df = ensemble_df.assign(ensemble_prediction = mode_of_predictions[0])
    ensemble_df[ACTUAL_COL_NAME] = actual_labels
    output_df = ensemble_df[[ACTUAL_COL_NAME, PREDICT_COL_NAME]]

    summary_metrics = compute_result_metrics(output_df, ACTUAL_COL_NAME, PREDICT_COL_NAME, classification_type)
    metrics_list.append(summary_metrics)

    global best_run_accuracy, best_run_prediction_data
    # Save data from the best result for plots.
    if summary_metrics['Accuracy'] > best_run_accuracy:
        best_run_accuracy = summary_metrics['Accuracy']
        best_run_prediction_data = output_df.rename(columns = {ACTUAL_COL_NAME: CLASS_COL, PREDICT_COL_NAME: 'probability'})

