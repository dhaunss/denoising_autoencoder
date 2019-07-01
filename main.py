#!/usr/bin/python3
import os
import sys
from itertools import product

import keras
import matplotlib.pyplot as plt
import numpy as np
from utils.plot import plot_traces, plot_history, plot_energy_distribution
from utils.processdata import processdata
from utils.utils import create_directory, load_data_w_sim_noise, shuffle_data, calculate_energy, calculate_signal2noise, \
	save_test_data_2_hdf5

#root_dir = "/cr/users/hanuss/data/"

root_dir=  "/home/dominik/workdir/datadir/"

os.environ["CUDA_VISIBLE_DEVICES"]="1"

####################set train para#####################



def fit_Regressor(para,root_dir ,output_directory):
	k,f,b,l,d, n_bins , noise_scale = para
	data = load_data_w_sim_noise(root_dir, scale=noise_scale)
	shuffle_data(data)

	process = processdata(data)
	process.shift_traces(nbins=n_bins)

	process.linear_normalisation()

	process.split_data()


	process.convert_data()

	x_train = process.x_train
	y_train = process.y_train
	x_test = process.x_test
	y_test = process.y_test

	save_test_data_2_hdf5(x_test)


	print(len(x_test), len(y_test))
	print(len(y_train), len(y_train))
	input_shape = (1000, 1, 1)

	Regressor = create_Regressor(Regressor_name, input_shape= input_shape, para= para, output_directory= output_directory)

	Regressor.fit(x_train, y_train, para, outfile=output_directory)

	model = keras.models.load_model(f"{output_directory}/best_model.hdf5")





	y_pred = model.predict(x_test)
	print(len(y_pred))
	print(len(y_test))
	y_pred = np.squeeze(y_pred)
	y_test = np.squeeze(y_test)
	x_test = np.squeeze(x_test)
	e_pred = calculate_energy(y_pred, process.shifts)
	e_real = calculate_energy(y_test, process.shifts)
	e_pred_snr = calculate_signal2noise(y_pred)
	e_real_snr = calculate_signal2noise(y_test)

	plt.hist([e_pred_snr,e_real_snr],bins=30, label=["SNR Prdediction","SNR True" ])
	plt.legend(loc='upper left')
	plt.xlim(0,150)
	plt.savefig(f"{output_directory}/SNR.png")
	plt.close()
	diff = (e_pred - e_real) / e_real
	snr_3 = []

	for x,y in zip(e_pred_snr, diff):
		if x > 3:
			snr_3.append(y)

	snr_2 = []
	for x,y in zip(e_pred_snr, diff):
		if x > 2:
			snr_2.append(y)


	history = np.genfromtxt(f"{output_directory}/history.csv", delimiter=',', names=True)
	plot_energy_distribution(snr_3, path_outfile=output_directory, label="snr>3")
	plot_energy_distribution(snr_2, path_outfile=output_directory,  label="snr>2")
	plot_energy_distribution(diff, path_outfile=output_directory , label=None)

	plot_history(history, output_directory)
	plot_traces(x_test, y_test,y_pred, output_directory, title="result")
	#plot_energy(e_pred, e_real, output_directory)


def create_Regressor(Regressor_name, input_shape, para, output_directory):
	if Regressor_name == 'aa':
		from Regressors import aa
		return aa.Regressor_aa(output_directory, para, input_shape)
	if Regressor_name == 'resnet':
		from Regressors import resnet
		return resnet.Regressor_RESNET ( output_directory, input_shape)
	if Regressor_name == 'cnn':  # Time-CNN
		from Regressors import cnn
		return cnn.Regressor_CNN ( output_directory, input_shape)


############################################### main
n_bins = [200]
noise_scale =[0.00055]
l2 = [0]
dropout= [0]
kernel = [5]
min_filter = [16]
batch_size = [128]


Regressor_name = sys.argv[1]

for x in product(kernel, min_filter, batch_size, l2,dropout, n_bins, noise_scale):

	output_directory = f"{root_dir}results/{str(Regressor_name)}/para{x}"
	create_directory(output_directory)

	print("made file on:", output_directory)

	print ( 'Method: ', Regressor_name)

	fit_Regressor(para=x, root_dir= root_dir, output_directory=output_directory)

	print('DONE')

