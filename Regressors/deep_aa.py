# Encoder
import keras
import numpy as np
import time
from keras import layers
from Regressors.model_util.model_util import encoding_layer, decoding_layer
from utils.plot import plot_history, plot_traces

class Regressor_deep_aa:

    def __init__(self, output_directory, input_shape):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape)
        self.callbacks = []

    def build_model(self, input_shape, kernal = 3):
        input_layer = keras.layers.Input(input_shape)
        ####autoencoder################

        # 1000 -> 1000
        x = encoding_layer(input_layer, nfilter=4, kernal=kernal)

        # 1000 -> 500
        x  = encoding_layer(x, nfilter=16, kernal=kernal, stride=2)
        x  = encoding_layer(x, nfilter=16, kernal=kernal, stride=1)
        x1 = encoding_layer(x, nfilter=16, kernal=kernal, stride=1)
        x1 = encoding_layer(x1, nfilter=16, kernal=kernal, stride=1)

        # 500 -> 250
        x = encoding_layer(x1, nfilter=32, kernal=kernal, stride=2)
        x = encoding_layer( x, nfilter=32, kernal=kernal, stride=1)
        x = encoding_layer( x, nfilter=32, kernal=kernal, stride=1)
        x = encoding_layer( x, nfilter=32, kernal=kernal, stride=1)

        # 250 -> 125
        x  = encoding_layer(x, nfilter=64, kernal=kernal, stride=2)
        x  = encoding_layer(x, nfilter=64, kernal=kernal, stride=1)
        x2 = encoding_layer(x, nfilter=64, kernal=kernal, stride=1)
        x2 = encoding_layer(x2, nfilter=64, kernal=kernal, stride=1)

        # 125 -> 25
        x = encoding_layer(x2, nfilter=128, kernal=kernal, stride=5)
        x = encoding_layer(x, nfilter=128, kernal=kernal, stride=1)

        # 25 -> 125
        x = decoding_layer(x, nfilter=64, kernal=kernal, stride=5, drop=0)
        x4 = decoding_layer(x, nfilter=64, kernal=kernal, stride=1, drop=0)
        sc2 = layers.Add()([x4, x2])

        # 125 -> 250
        x = decoding_layer(sc2, nfilter=32, kernal=kernal, stride=2, drop=0)
        x = decoding_layer(x, nfilter=32, kernal=kernal, stride=1, drop=0)

        # 250 -> 500
        x = decoding_layer(x, nfilter=16, kernal=kernal, stride=2, drop=0)
        x3 = decoding_layer(x, nfilter=16, kernal=kernal, stride=1, drop=0)
        sc1 = layers.Add()([x3, x1])

        # 500 -> 1000
        x = decoding_layer(sc1, nfilter=8, kernal=kernal, stride=2, drop=0)
        xout = decoding_layer(x, nfilter=1, kernal=kernal, stride=1, drop=0)

        output_layer = layers.Add()([xout, input_layer])

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mse', optimizer=keras.optimizers.Adam())

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)

        file_path = f"{self.output_directory}best_model.hdf5"

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        print(model.summary())
        return model

    def fit(self, x_train, y_train, x_test, y_test, batch_size= 128, ):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        nb_epochs =   2

        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                                                validation_split = 0.1, callbacks=self.callbacks)

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_test)

        plot_history(hist, self.output_directory)
        plot_traces(x_test, y_test, y_pred, self.output_directory)




        keras.backend.clear_session()