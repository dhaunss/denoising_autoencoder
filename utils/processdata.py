import numpy as np
from utils.utils import array_shifting, get_highest_amplitudes, calculate_energy


class processdata:
	def __init__(self, data):
		self.noisy_data = data["input_traces"]
		self.label_data = data["label_traces"]

		self.dim = 1
		self.tl = 1000
		self.norm = []

		self.shifts = []
		self.norm = []

	def split_data(self):
		# ------- Set and process data -------
		n = self.noisy_data.shape[0]
		n_split = n // 10

		self.x_train, self.x_test = np.split(self.noisy_data, [n_split * 9])
		self.y_train, self.y_test = np.split(self.label_data, [n_split * 9])
		#self.e_train, self.e_test = np.split(self.energy, [n_split * 9])

	def split_to_test(self):
		n = self.noisy_data.shape[0]
		n_split = n // 10

		self.noisy_data = self.noisy_data[:n_split]
		self.label_data = self.label_data[:n_split]

	def linear_normalisation(self):
		self.norm = 1. / get_highest_amplitudes(self.noisy_data)

		self.noisy_data *= self.norm[..., None]
		self.label_data *= self.norm[..., None]

	def convert_data(self):
		# Function to actually init data feed in network
		self.x_train = self.convert_sequence(self.x_train)
		self.y_train = self.convert_sequence(self.y_train)
		self.x_test  = self.convert_sequence(self.x_test)
		self.y_test  = self.convert_sequence(self.y_test)
		#self.e_train = self.convert_sequence(self.e_train)
		#self.e_test = self.convert_sequence(self.e_test)



	def convert_sequence(self, sequence):
		lenth = len(sequence)
		return np.reshape(sequence, (lenth, self.tl, self.dim, 1))

	def shift_traces(self, nbins):
		self.noisy_data, self.label_data, self.shifts = array_shifting(self.noisy_data,
																	   self.label_data, nbins)

	def re_normalize_traces(self):
		self.x_test/=self.norm[...,None]
		self.y_test/=self.norm[...,None]


	def choose_data(self, test):
		return self.test_data if test else self.train_data

	def get_energy(self):
		self.energy = calculate_energy(self.label_data, self.shifts)


