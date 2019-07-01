# Encoder
import keras
import numpy as np
import time
from keras import layers
from Regressors.model_util.model_util import encoding_layer, decoding_layer
from utils.plot import plot_history, plot_traces
from utils.utils import create_directory
from utils.processdata import processdata


class Regressor_aa:

    def __init__(self, output_directory,para, input_shape):
        self.output_directory = output_directory
        print(self.output_directory)
        create_directory(self.output_directory)

        print(self.output_directory)

        self.model = self.build_model(input_shape, para)


    def build_model(self, input_shape, para):

        kernel, max_filter , batch_size , regularization = para
        min_filter = max_filter/8




        input_layer = keras.layers.Input(input_shape)
        ####autoencoder################




        # 1000 -> 1000
        x = encoding_layer(input_layer, nfilter=4, kernal=5)

        # 1000 -> 500
        x = encoding_layer(x, nfilter=min_filter, kernal=kernel, stride=2, regularize=regularization)
        x1 = encoding_layer(x, nfilter=min_filter, kernal=kernel, stride=1, regularize=regularization)

        # 500 -> 250
        x = encoding_layer(x1, nfilter=min_filter*2, kernal=kernel, stride=2, regularize=regularization)
        x = encoding_layer(x, nfilter=min_filter*2, kernal=kernel, stride=1, regularize=regularization)

        # 250 -> 125
        x = encoding_layer(x, nfilter=min_filter*4, kernal=kernel, stride=2, regularize=regularization)
        x2 = encoding_layer(x, nfilter=min_filter*4, kernal=kernel, stride=1, regularize=regularization)

        # 125 -> 25
        x = encoding_layer(x2, nfilter=min_filter*8, kernal=kernel, stride=5, regularize=regularization)
        x = encoding_layer(x, nfilter=min_filter*8, kernal=kernel, stride=1,regularize=regularization)

        # 25 -> 125
        x = decoding_layer(x, nfilter=min_filter*4, kernal=kernel, stride=5, drop=0,regularize=regularization)
        x4 = decoding_layer(x, nfilter=min_filter*4, kernal=kernel, stride=1, drop=0,regularize=regularization)
        sc2 = layers.Add()([x4, x2])

        # 125 -> 250
        x = decoding_layer(sc2, nfilter=min_filter*2, kernal=kernel, stride=2, drop=0, regularize=regularization)
        x = decoding_layer(x, nfilter=min_filter*2, kernal=kernel, stride=1, drop=0, regularize= regularization)

        # 250 -> 500
        x = decoding_layer(x, nfilter=min_filter, kernal=kernel, stride=2, drop=0, regularize=regularization)
        x3 = decoding_layer(x, nfilter=min_filter, kernal=kernel, stride=1, drop=0, regularize=regularization)
        sc1 = layers.Add()([x3, x1])

        # 500 -> 1000
        x = decoding_layer(sc1, nfilter=min_filter/2, kernal=kernel, stride=2, drop=0,regularize=regularization)
        xout = decoding_layer(x, nfilter=1, kernal=kernel, stride=1, drop=0,regularize=regularization)

        output_layer = layers.Add()([xout, input_layer])

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mse', optimizer=keras.optimizers.Adam())


        print(model.summary())
        return model

    def fit(self, x_train, y_train, outfile, para):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        nb_epochs =  1
        kernel, max_filter , batch_size , regularization = para

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
        logger = keras.callbacks.CSVLogger(f"{outfile}/history.csv")
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=f"{outfile}/best_model.hdf5",
                                                           monitor='loss',
                                                           save_best_only=True)

        callbacks =[reduce_lr, model_checkpoint, logger]
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                                                validation_split = 0.2, callbacks=callbacks)


        keras.backend.clear_session()