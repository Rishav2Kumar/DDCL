'''
Created on Dec 5, 2022

Subclass of Keras model created for logging custom metrics at the iteration (batch) level.
'''

from datetime import datetime
import tensorflow as tf

log_dir = './Results/batch_level/' + datetime.now().strftime('%Y-%m-%d_%H%M_%S')


class NeuralNetwork(tf.keras.Model):
    '''
    classdocs
    '''

    def __init__(self, model):
        '''
        Constructor
        '''
        super().__init__()
        self.model = model

    def train_step(self, data):
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
