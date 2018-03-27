import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras import backend as K
K.set_image_data_format('channels_first')


class Classifier(object):
    """Plain-vanilla classifier

    Inputs
    ------
    input_shape : list-like of int (length 2 or 3)
        - shape of each input image. e.g. `(3, 50, 50)`

    num_conv_filters: Optional(list-like of int)
         - List of `filters` parameters for `keras.layers.convolutional.Conv2D`
         - Length of this parameter should match the number of conv layers desired

    conv_kernel_sizes: Optional(list-like of int)
         - List of `kernel_size` parameters for `keras.layers.convolutional.Conv2D`
         - Length of this parameter should match the number of conv layers desired

    padding: Optional(str)
        - parameter for `keras.layers.convolutional.Conv2D`

    pool_size: Optional(ist-like of int [length 2])
        - the parameter for `keras.layers.convolutional.MaxPooling2D`

    dropout_fraction: Optional(float)
        - parameter for `keras.layers.core.Dropout` after each conv layer

    activation: Optional(str)
        - parameter for `keras.layers.core.Activation` between each layer
          (but not after the final dense layer)

    activation_final: Optional(str)
        - the final activation of the output layer

    num_neurons_dense: Optional(list-like of int)
         - Number of dense layers set by length of this parameter
         - `num_neurons_dense[-1]` sets the number of outputs from the network

    loss_function_name: Optional(str)
        - the loss function which the optimizer tries to minimize

    batch_size: Optional(int)
        - batch the training into `batch_size` amount of images

    """
    def __init__(self,
                 input_shape,
                 num_conv_filters=(16, 16, 16),
                 conv_kernel_sizes=(4, 8, 16),
                 padding="same",
                 pool_size=(2, 2),
                 dropout_fraction=0.25,
                 activation="relu",
                 activation_final="sigmoid",
                 num_neurons_dense=(128, 64, 1),
                 loss_function_name="binary_crossentropy",
                 batch_size=64):

        assert(len(num_conv_filters) == len(conv_kernel_sizes))

        self.input_shape = input_shape
        self.num_conv_filters = num_conv_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.padding = padding
        self.pool_size = pool_size
        self.dropout_fraction = dropout_fraction
        self.activation = activation
        self.activation_final = activation_final
        self.num_neurons_dense = num_neurons_dense
        self.loss_function_name = loss_function_name
        self.batch_size = batch_size

    def configure_optimizer(self, lr=0.001, **kwargs):
        """Configure an `Adam` optimizer
        """
        kwargs["lr"] = lr
        self.optimizer = Adam(**kwargs)

    def configure_early_stopping(self,
                                 monitor="loss",
                                 patience=35,
                                 verbose=1,
                                 mode="auto",
                                 **kwargs):
        """Configure patience-based early stopping on our optimizer
        """
        self.early_stopping = EarlyStopping(monitor=monitor,
                                            patience=patience,
                                            verbose=verbose,
                                            mode=mode,
                                            **kwargs,
                                            )

    def build_model(self):
        model = Sequential()

        # # # Convolutional Layers
        num_conv_layers = len(self.num_conv_filters)
        for i in range(num_conv_layers):
            model.add(Conv2D(self.num_conv_filters[i],
                             self.conv_kernel_sizes[i],
                             padding=self.padding,
                             input_shape=self.input_shape,
                             ))
            model.add(Activation(self.activation))
            model.add(MaxPooling2D(pool_size=self.pool_size))
            model.add(Dropout(self.dropout_fraction))

        model.add(Flatten())

        # # # Dense / Fully-Connected Layers
        number_of_dense_layers = len(self.num_neurons_dense)
        for i, num_neurons in enumerate(self.num_neurons_dense):
            if (i+1) != number_of_dense_layers:
                activation = self.activation
            else:
                activation = self.activation_final
            model.add(Dense(num_neurons, activation=activation))

        model.compile(loss=self.loss_function_name,
                      optimizer=self.optimizer)

        self.model = model

    def fit_model(self,
                  X,
                  Y,
                  training_set_indices,
                  testing_set_indices,
                  data_iterator,
                  verbose=1,
                  max_epochs=100):

        batches_per_epoch = training_set_indices.size//self.batch_size

        if verbose:
            print("batches_per_epoch: ", batches_per_epoch)
            print("batch_size: ", self.batch_size)

        history = self.model \
                      .fit_generator(data_iterator,
                                     steps_per_epoch=batches_per_epoch,
                                     epochs=max_epochs,
                                     validation_data=(X[testing_set_indices],
                                                      Y[testing_set_indices]),
                                     verbose=verbose,
                                     callbacks=[self.early_stopping],
                                     )

        return history
