from iqtools.tools import read_rsa_specan_xml, read_rsa_data_csv, read_rsa_result_csv
import numpy as np
import os

def read_tdsm_bin(path):
    base_path, _ = os.path.splitext(path)
    bin_fre_path = os.path.join(base_path + '.bin_fre')
    bin_time_path = os.path.join(base_path + '.bin_time')
    bin_amp_path = os.path.join(base_path + '.bin_amp')

    try:
        fre = np.fromfile(bin_fre_path, dtype=np.float64)
        time = np.fromfile(bin_time_path, dtype=np.float32)
        amp = np.fromfile(bin_amp_path, dtype=np.float32)
    except IOError as e:
        raise Exception(f"Error reading files: {e}")
    
    if len(time) == 0 or len(fre) == 0:
        raise ValueError("Time or frequency data files are empty")
    
    try:
        amp = amp.reshape((len(time), len(fre)))
    except ValueError as e:
        raise ValueError(f"Amplitude data cannot be reshaped: {e}")

    midpoint = len(fre) // 2
    frequency = np.concatenate((fre[midpoint:], fre[:midpoint]))
    amplitude = np.concatenate((amp[:, midpoint:], amp[:, :midpoint]), axis=1)

    return frequency, time, amplitude

def handle_read_tdsm_bin(path):
    frequency, _, amplitude = read_tdsm_bin(path)
    amplitude_avg = np.average(amplitude, axis = 0)
    return frequency, amplitude_avg

def handle_read_rsa_specan_xml(filename):
    f, p, _ = read_rsa_specan_xml(filename)
    return f, p

def handle_read_rsa_data_csv(filename):
    data = read_rsa_data_csv(filename)
    #obtain frequency, power from data
    return data

def handle_read_rsa_result_csv(filename):
    f, p = read_rsa_result_csv(filename)
    return f, p

def read_psdata(filename, dbm = False):
    if dbm: return np.genfromtxt(filename, skip_header = 1, delimiter='|', usecols = (0,2))
    else: return np.genfromtxt(filename, skip_header = 1, delimiter='|', usecols = (0,1))
