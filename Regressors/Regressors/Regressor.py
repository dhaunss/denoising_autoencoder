import numpy as np
import keras
import copy

from utilities import tools, analysis, process_data

from NNTrainer import Trainer
#import NNTrainer.model_utilities.losses as loss



class TimeTraceRegressor(Trainer.Trainer):
    def __init__(self, *args, **kwargs):
        super(TimeTraceRegressor, self).__init__(*args, **kwargs)

        self.loss.append("mse")

        self.dim = 1
        self.tl = 1000

        self.normalisation = process_data.linear_normalisation

        self.callbacks = {"EarlyStopping": keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=2, min_delta=0.00001),
                          "ReduceLROnPlateau": keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=2, min_lr=0.00001)}

    def set_data(self, data):
        # ------- Set and process data -------
        n = data["input_traces"].shape[0]
        n_split = n // 10

        traces_train, traces_test = np.split(data["input_traces"], [n_split * 9])
        label_traces_train, label_traces_test = np.split(data["label_traces"], [n_split * 9])
        label_classes_train, label_classes_test = np.split(data["label_classes"], [n_split * 9])
        identifier_train, identifier_test = np.split(data["identifier"], [n_split * 9])

        self.train_data = {"traces": traces_train, "label_traces": label_traces_train, "label_classes": label_classes_train, "identifier": identifier_train}
        self.test_data = {"traces": traces_test, "label_traces": label_traces_test, "label_classes": label_classes_test, "identifier": identifier_test}

    def init_data(self):
        # Function to actually init data feed in network
        self.set_train_input_data(self.convert_sequence(self.train_data["traces"]))
        self.set_train_input_label(self.convert_sequence(self.train_data["label_traces"]))
        self.set_test_input_data(self.convert_sequence(self.test_data["traces"]))
        self.set_test_input_label(self.convert_sequence(self.test_data["label_traces"]))

    def set_parameter(self, para):
        # Should be defined in sub classes ...
        # Defaults set in trainer classes
        self.epochs = para.epoch or self.epochs
        self.batch = para.batch or self.batch
        self.learning_rate = para.learning_rate or self.learning_rate

    def normalize_traces(self, data):
        if self.normalisation is not None:
            data["traces"], data["label_traces"], data["norm"] = self.normalisation(data)
        else:
            data["norm"] = np.squeeze(np.ones((len(data["traces"]), self.dim)))

    def shift_traces(self, data, Nbins):
        data["traces"], data["label_traces"], data["shifts"] = tools.array_shifting(data["traces"], data["label_traces"], Nbins)

    def re_normalize_traces(self, test=0):
        data = self.choose_data(test)
        for key in ["traces", "label_traces", "rec_traces"]:
            data[key] /= data["norm"][..., None]

    def get_prediction(self):
        results = np.squeeze(self.keras_model.predict(self._test_input_data))
        self.test_data["rec_traces"] = results

    def get_prediction_train(self):
        results = np.squeeze(self.keras_model.predict(self._train_input_data))
        self.train_data["rec_traces"] = results

    def convert_sequence(self, sequence):
        return np.reshape(sequence, (sequence.shape[0], self.tl, self.dim, 1))

    def calculate_snr(self, label=1):
        data = self.choose_data(test=1)
        for key in ["traces", "rec_traces", "label_traces"]:
            data["snr_" + key] = analysis.calculate_signal2noise(data[key], data["shifts"])

    def calculate_energy(self, label=1):
        data = self.choose_data(test=1)
        # mask = np.array(data["label_classes"] == label)
        for key in ["traces", "rec_traces", "label_traces"]:
            data["energy_" + key] = analysis.calculate_energy(data[key], data["shifts"])

