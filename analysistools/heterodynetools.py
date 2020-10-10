import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import e6py as e6
import h5py
import xarray as xr
import specarray as sa

def butter_lowpass_filter(data, cutoff_freq, sample_freq, order):
    nyquist_freq = sample_freq / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    # Get the filter coefficients
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def demodulate(het_xr,carrier_freq, bw):
    time_data = het_xr.coords['time'].values
    LO_signal = np.exp(1j * 2 * np.pi * carrier_freq * time_data)
    demod_sig = het_xr * LO_signal
    demod_sig = butter_lowpass_filter(demod_sig, bw, 80e6, 3)
    I_sig = np.real(demod_sig)
    Q_sig = np.imag(demod_sig)
    A_sig = np.sqrt(I_sig**2 + Q_sig**2)
    phi_sig = np.arctan2(Q_sig, I_sig)
    I_xr = xr.DataArray(I_sig, dims=['time'], coords={'time':time_data})
    Q_xr = xr.DataArray(Q_sig, dims=['time'], coords={'time':time_data})
    A_xr = xr.DataArray(A_sig, dims=['time'], coords={'time':time_data})
    phi_xr = xr.DataArray(phi_sig, dims=['time'], coords={'time':time_data})
    return I_xr, Q_xr, A_xr, phi_xr