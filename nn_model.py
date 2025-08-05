'''
Created on Dec 5, 2022

Subclass of Keras model created for logging custom metrics at the iteration (batch) level.
'''

from datetime import datetime
import tensorflow as tf

log_dir = './Results/batch_level/' + datetime.now().strftime('%Y-%m-%d_%H%M_%S')


class NeuralNetwork(tf.keras.Model):
    '''
    A custom Keras Model wrapper for neural networks.
    This class encapsulates a Keras model and customizes the training step logic, allowing for fine-grained control over the training process. It supports custom loss computation, gradient calculation, optimizer updates, and metric tracking.
        model (tf.keras.Model): The underlying Keras model to be wrapped.
    Methods:
        train_step(data):
            Performs a single training step on a batch of data, including forward pass, loss computation, gradient calculation, weight updates, and metric tracking.
        call(x):
            Invokes the underlying model on input data `x`.
    '''

    def __init__(self, model):
        '''
        Constructor
        '''
        super().__init__()
        self.model = model

    def train_step(self, data):
        """
        Performs a single training step on a batch of data.
        Args:
            data (tuple): A tuple (x, y) where x is the input data and y is the target labels.
        Returns:
            dict: A dictionary mapping metric names to their current values after the training step.
        This method computes the forward pass, calculates the loss, computes gradients, updates model weights,
        and updates the tracked metrics for the current batch.
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training = True)
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses = self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)

        metric_dict = {m.name: m.result() for m in self.metrics}
        return metric_dict

    def call(self, x):
        x = self.model(x)
        return x
