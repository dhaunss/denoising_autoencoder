import numpy as np
import h5py

from utils.processdata import processdata
from utils.utils import create_directory, load_data_w_sim_noise, shuffle_data

root_dir=  "/home/dominik/workdir/datadir/"


data = load_data_w_sim_noise(root_dir, scale=0.00055)

shuffle_data(data)

process = processdata(data)

process.shift_traces(nbins=100)

process.linear_normalisation()

process.split_data()


x_signal = process.x_test
y_signal = process.y_test



f = h5py.File("input_signal.hdf5", "w")
signal_ds = f.create_dataset("input_signal", data=x_signal)
f.close()

f = h5py.File("output_signal.hdf5", "w")
signal_ds = f.create_dataset("output_signal", data=y_signal)
f.close()