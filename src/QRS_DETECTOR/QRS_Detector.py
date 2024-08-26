import numpy as np
from scipy.signal import butter, filtfilt

def qrs_detector(sig, freq_sampling):
    cleaned_ecg = preprocess_ecg(sig, freq_sampling, 5, 22, size_window = int( 0.1 * freq_sampling))
    peaks = detect_peaks(cleaned_ecg, no_peak_distance= int(freq_sampling*0.65), distance = int(freq_sampling * 0.33))
    qrs_indices = threshold_detection(cleaned_ecg, peaks, freq_sampling, initial_search_samples= int(freq_sampling * 0.83), long_peak_distance=int(freq_sampling*1.111))
    return qrs_indices

def detect_peaks(cleaned_ecg, no_peak_distance, distance=0):
    last_max = -np.inf  # The most recent encountered maximum value
    last_max_pos = -1  # Position of the last_max in the array
    peaks = [np.argmax(cleaned_ecg[:no_peak_distance])]  # Detected peaks positions
    peak_values = [cleaned_ecg[peaks[0]]]  # Detected peaks values

    
    for i, current_value in enumerate(cleaned_ecg):
       
        # Update the most recent maximum if the current value is greater
        if current_value > last_max:
            last_max = current_value
            last_max_pos = i
        
        # Check if the current value is less than half the last max
        # or if we are beyond the no_peak_distance from the last max
        if current_value <= last_max / 2 or (i - last_max_pos >= no_peak_distance):
            # Check if the last peak is within the `distance` of the current peak
            if last_max_pos - peaks[-1] < distance:
                # If within the distance, choose the higher peak
                if last_max > peak_values[-1]:
                    peaks[-1] = last_max_pos
                    peak_values[-1] = last_max
            else:
                # Otherwise, start a new peak group
                peaks.append(last_max_pos)
                peak_values.append(last_max)
            
            # Reset the last max after adding a peak
            last_max = current_value
            last_max_pos = i
    
    return np.array(peaks)

def threshold_detection(cleaned_ecg, peaks, fs, initial_search_samples=300, long_peak_distance=400):

    spk = 0.13 * np.max(cleaned_ecg[:initial_search_samples])
    npk = 0.1 * spk
    threshold = 0.25 * spk + 0.75 * npk
    
    qrs_peaks = []
    noise_peaks = []
    qrs_buffer = []
    last_qrs_time = 0
    min_distance = int(fs * 0.12)
    
    for i, peak in enumerate(peaks):
        peak_value = cleaned_ecg[peak]
        
        if peak_value > threshold:
            if qrs_peaks and (peak - qrs_peaks[-1] < min_distance):
                if peak_value > cleaned_ecg[qrs_peaks[-1]]:
                    qrs_peaks[-1] = peak
            else:
                qrs_peaks.append(peak)
                last_qrs_time = peak
            
            spk = 0.25 * peak_value + 0.75 * spk
            
            qrs_buffer.append(peak)
            if len(qrs_buffer) > 10:
                qrs_buffer.pop(0)
        else:
            noise_peaks.append(peak)
            npk = 0.25 * peak_value + 0.75 * npk
        
        threshold = 0.25 * spk + 0.75 * npk
        
        if peak - last_qrs_time > long_peak_distance:
            spk *= 0.5
            threshold = 0.25 * spk + 0.75 * npk
            for lookback_peak in peaks[i-5:i+1]:
                if lookback_peak != last_qrs_time:
                    if last_qrs_time < lookback_peak < peak and cleaned_ecg[lookback_peak] > threshold:
                        qrs_peaks.append(lookback_peak)
                        spk = 0.875 * spk + 0.125 * cleaned_ecg[lookback_peak]
                        threshold = 0.25 * spk + 0.75 * npk
                        last_qrs_time = lookback_peak
                        break
        
        if len(qrs_buffer) > 1:
            rr_intervals = np.diff(qrs_buffer)
            mean_rr = np.mean(rr_intervals)
            if peak - last_qrs_time > 1.5 * mean_rr:
                spk *= 0.5
                threshold = 0.25 * spk + 0.75 * npk
    
    return np.array(qrs_peaks)

def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='high')
    y = filtfilt(b, a, data)
    return y

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    b, a = butter(order, low, btype='low')
    y = filtfilt(b, a, data)
    return y

def differentiate(data):
    return np.diff(data, prepend=data[0])

def squaring(data):
    return np.square(data)

def moving_window_integration(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def preprocess_ecg(data, fs, high, low, size_window):
    signal = highpass_filter(data, high, fs)
    signal = lowpass_filter(signal, low, fs)
    signal = differentiate(signal)
    signal = squaring(signal)
    signal = moving_window_integration(signal, size_window)
    return signal