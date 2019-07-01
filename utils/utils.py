import os
import sys

import h5py
import numpy as np
from scipy.signal import hilbert


def load_npz_data(datadir=None, fname=None):
    datadir = datadir or "/home/arg/workdir/datadir/"
    fname = fname or "hd5_data_lpda.npz"

    print("Reading %s%s ..." % (datadir, fname))
    data = np.load(datadir + fname)

    # data_dir = utils.convert_npz_to_dirs(data)

    data_dir = {
        "input_traces": data["signal_noise"],
        "label_traces": data["signal"],
        "label_classes": np.ones(len(data["signal"])),
        "stations": data["stations"],
        "channels": data["channels"],
        "identifier": data["identifier"]
    }

    return data_dir


def get_and_scale_sim_noise(datadir, scale, spread=0.2):
    noise_traces = np.load(f'{datadir}simulated_noise_traces.npz', "r")["noise_traces"]

    # Scale simulated noise
    # first normalize mean max amplitude to 1 and then multiply with scale
    print("Scale sim noise to: %f" % scale)
    noise_traces *= 1. / np.mean(np.amax(np.abs(noise_traces), axis=1)) * scale

    # # increas spread of rms
    # print("multiply smearing of %f" % spread)
    # noise_traces *= np.random.normal(loc=1., scale=spread, size=len(noise_traces))[:, None]

    return noise_traces


def load_data_w_sim_noise(datadir=None, fname=None, scale=0.00055):
    datadir = datadir or"/home/arg/workdir/datadir/"
    fname = fname or "hd5_data_lpda.npz"

    print("Reading %s%s ..." % (datadir, fname))
    data = np.load(datadir + fname, "r")

    noise_traces = get_and_scale_sim_noise(datadir, scale)

    noise_traces = noise_traces[:len(data["signal"])]

    data_dir = {
        "input_traces": noise_traces + data["signal"],
        "label_traces": data["signal"],
        "label_classes": np.ones(len(data["signal"])),
        "stations": data["stations"],
        "channels": data["channels"],
        "identifier": data["identifier"]
    }

    return data_dir


def cut_traces_to_region(traces, regions):
    if traces.ndim == 1:
        cut_traces = traces[regions]
    elif traces.ndim == 2:
        cut_traces = np.array([[traces[i, j] for j in regions[i]] for i in range(len(traces))])
    elif traces.ndim == 3:
        cut_traces = np.array([[[traces[i, j, k] for k in regions[i]] for j in range(traces.shape[1])] for i in range(len(traces))])
    elif traces.ndim == 4:  # special case for evaluationg in callback
        cut_traces = np.array([[np.squeeze(traces)[i, j] for j in regions[i]] for i in range(len(traces))])

    return np.squeeze(cut_traces)


def calculate_energy(traces, shifts=None, use_hilbert=False):
    # Calculates the energy in voltage time traces
    # to be able to proceed with 1 and 2d arrays
    if traces.ndim == 1:
        traces = np.expand_dims(traces, axis=0)

    if shifts is None:
        shifts = np.zeros(len(traces), dtype=int)

    # Get indices for shifted signal/noise region
    signal_region = get_shifted_region(shifts, 400, 600)
    noise_region = np.hstack((get_shifted_region(shifts, 200, 400), get_shifted_region(shifts, 600, 800)))

    if use_hilbert:
        traces = np.abs(hilbert(traces, axis=-1))

    # array can be either 1d or 2d
    voltage_square = np.square(traces)
    signal = cut_traces_to_region(voltage_square, signal_region)
    noise = cut_traces_to_region(voltage_square, noise_region)

    R = 50.  # Ohm
    dt = 1. / 180e6 * 1e9  # 1/180MHz in nanoseconds
    e = 1.6021766208e-19
    integrals_sum = (np.sum(signal, axis=-1) * dt - 1/2. * np.sum(noise, axis=-1) * dt) / R
    energy = integrals_sum * 1e-9 / e  # eV (correcting ns)

    return energy


def get_highest_amplitudes(traces, ranges=None, use_hilbert=False):
    if ranges is None:
        selected = traces
    else:
        selected = cut_traces_to_region(traces, ranges)

    if use_hilbert:
        envolops = np.abs(hilbert(selected, axis=-1))
        has = np.amax(envolops, axis=-1)
    else:
        has = np.amax(np.abs(selected), axis=-1)

    return has


def array_shifting(arr1, arr2, nbins):
    # Shifting the signal forth and back. 36 Bins corresponse with a bin width if 5.5 ns to a window of ca. 200 ns
    # In offline, SD depending signal searchwindow, -150, +100 ns (+ core, direction uncertainty)
    # But on efield trace signals are narrow, so changed do conservative 200 bins +-
    if arr1.size != arr2.size:
        sys.exit("Shifting failed, arrays do not have the same size")

    random_shifts = np.random.randint(-nbins, nbins, len(arr1))

    arr1 = np.array([np.roll(arr1[idx], random_shifts[idx]) for idx in range(len(arr1))])
    arr2 = np.array([np.roll(arr2[idx], random_shifts[idx]) for idx in range(len(arr2))])

    return arr1, arr2, random_shifts


def shift_traces(trace_1, trace_2, nbins):
    random_shift = np.random.randint(-nbins, nbins)
    return np.roll(trace_1, random_shift), np.roll(trace_2, random_shift), random_shift


def shift_traces_back(data_trace, label_trace, shifts):
    return np.roll(data_trace, -1 * shifts), np.roll(label_trace, -1 * shifts)


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def shuffle_data(data, seed=None):
    np.random.seed(seed)
    # python 3.5: [*dict] -> list(dict_keys)
    N = len(data[list(data)[0]])
    randomize = np.arange(N)
    np.random.shuffle(randomize)

    for key in data:
        data[key] = data[key][randomize]


def get_shifted_region(shifts, low, up):
    return (np.arange(low, up, 1)[None, ...] + shifts[..., None]) % 1000


def plot_energy_deviation(en_test, en_signal, name, label=None, title=""):

    # title = "Energy deviation: label vs. %s " % name

    delta_eng = analysis.calculate_energy_deviation(en_test, en_signal)
    # schleife_energy = analysis.calculate_skewness(delta_eng)

    plt.rc('axes', labelsize=35)    # fontsize of the tick labels
    plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=25)    # fontsize of the tick labels

    fig, ax = php.get_histogram(
        delta_eng, bins=np.arange(-1.5, 2.01, 0.1), integral=[-0.8, 0.8], xlabel=r"$\frac{E_{%s}-E_{true}}{E_{true}}$" % name,
        # ylabel="Entries", title=title, stat_kwargs={"ha": "right", "fontsize": 25, "posx": 0.95, "additional_text": "Skewness: %.2f " % schleife_energy}, kwargs={'alpha': 1, 'facecolor': 'r'})
        ylabel="Entries", stat_kwargs={"ha": "right", "fontsize": 25, "posx": 0.98, "posy": 0.98},
        figsize=(12, 10))

    ax.set_title(title, fontsize=30)
    ax.axvline(-0.8, linestyle="--", color="k", alpha=0.6)
    ax.axvline(0.8, linestyle="--", color="k", alpha=0.6)
    # ax.axvline(0, linestyle="--", linewidth=1, color="r")

    plt.savefig(("energy_distr_%s.png" % name), bbox_inches='tight')
    plt.close(fig)


def calculate_signal2noise(traces, shifts=None, squared=False, use_hilbert=False):

    if shifts is None:
        shifts = np.zeros(len(traces), dtype=int)

    if traces.ndim == 1:
        traces = np.expand_dims(traces, axis=0)

    if shifts.ndim > 1:
        sys.exit("Shifts array has more than one dim!")

    # can be narrow as only max amp of signal is ask for
    signal_regions = get_shifted_region(shifts, 468, 568)
    noise_region = np.hstack((get_shifted_region(shifts, 300, 400), get_shifted_region(shifts, 600, 700)))
    highest_amplitudes = get_highest_amplitudes(traces, signal_regions, use_hilbert)

    noise_rms = calculate_rms(traces, noise_region)

    if highest_amplitudes.shape != noise_rms.shape:
        sys.exit("Shapes do not match!")

    snrs = np.where(noise_rms > 0, highest_amplitudes / noise_rms, -1)

    # The squared ratio max amplitude / rms noise is the offline default
    if squared:
        snrs = np.sign(snrs) * np.square(snrs)

    return snrs



def save_test_data_2_hdf5(data):
    hf = h5py.File("test.h5", "w")
    hf.create_dataset("test_input", data=data)
    hf.close()