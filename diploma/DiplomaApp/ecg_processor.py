import scipy.io
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=50, fs=500, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def load_ecg(file_path, seq_length=5000):
    mat_data = scipy.io.loadmat(file_path)
    ecg_signal = mat_data['val'][0][:seq_length]
    ecg_signal = butter_lowpass_filter(ecg_signal)
    return ecg_signal
