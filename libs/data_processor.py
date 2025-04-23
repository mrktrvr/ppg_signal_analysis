import os

import numpy as np
import pandas as pd

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


class DataProcessor:

    def __init__(self, **args):
        '''
        @Args
        fs (float): sampling frequency, Default 75, nyq = 0.5 * fs
        bandpass_lowcut (float): lowcut of bandpass, nyq * lowcut
        bandpass_highcut (float): highcut of bandpass, nyq * highcut
        bandpass_order (int): order ob bandpass filter
        peak_distance_factor (float): factor of find_peak distance
                                      fs * peak_distance_factor
        peak_prominence_factor (float): factor of find_peak prominence
                                        np.std(signal) * peak_prominence_factor
        uniform_filter_size (int): size of uniform_filter1d
        savgol_win_len (float): savgol filter window length
        savgol_polyorder (float): savgol filger polyorder
        '''
        self.fs = args.get('fs', 75)
        self.bandpass_lowcut = args.get('bandpass_lowcut', 0.3)
        self.bandpass_highcut = args.get('bandpass_highcut', 2.5)
        self.bandpass_order = args.get('bandpass_order', 5)
        self.peak_distance_factor = args.get('peak_distance_factor', 0.4)
        self.peak_prominence_factor = args.get('peak_prominence_factor', 0.001)
        self.peak_height_factor = args.get('peak_height_factor', 0.2)
        self.uniform_filter_size = args.get('uniform_filter_size', 30)
        self.savgol_win_len = args.get('savgol_win_len', 0)
        self.savgol_polyorder = args.get('savgol_polyorder', 5)

        self.channels = ['Red', 'Green', 'Blue']
        self.metric_names = ['HR', 'rMSSD', 'SDNN', 'pNN50']

    def bandpass_filter(self, signal, lowcut=None, highcut=None, order=None):
        '''
        Apply a Butterworth bandpass filter to isolate PPG frequency range.
        Cleans up the signal by applying a bandpass filter
        that keeps only the frequencies (lowcat to highcut).

        @Args
        signal (np.array): Input signal
        lowcut (float): Low cutoff frequency (Hz)
        highcut (float): High cutoff frequency (Hz)

        @Returns
        np.array: Filtered signal
        '''
        lowcut = self.bandpass_lowcut if lowcut is None else lowcut
        highcut = self.bandpass_highcut if highcut is None else highcut
        order = self.bandpass_order if order is None else order
        # --- Nyquist frequency (half the sampling rate)
        nyq = 0.5 * self.fs
        # --- Normalised low cutoff frequency
        low = lowcut / nyq
        # --- Normalised high cutoff frequency
        high = highcut / nyq
        # --- 2nd-order Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')
        # --- Apply the filter with zero-phase distortion
        ret = filtfilt(b, a, signal)
        return ret

    def _sig_reg_liner_interp(self, org_times, org_sig):
        '''
        Interpolate signal to regular time intervals using linear interpolation

        @Args
        org_times (pd.Series): Original timestamps
        org_signal (pd.Series): Original signal

        @Returns:
        new_time (np.array): regularised timestamps
        new_signal (np.array): interpolated signal
        '''
        step = 1 / self.fs
        new_times = np.arange(org_times[0], org_times[-1], step)
        prms = dict(
            # kind='linear',
            kind='cubic',
            fill_value='extrapolate',
        )
        interp_func = interp1d(org_times, org_sig, **prms)
        new_signal = interp_func(new_times)
        return new_times, new_signal

    def _sig_reg_skip_gaps(self, org_times, org_sig):
        new_sig = org_sig
        step = 1 / self.fs
        dlt = np.diff(org_times, prepend=-step)
        gap_idx = np.where(dlt > step)[0]
        new_times = np.copy(org_times)
        for i in gap_idx:
            new_times[i:] -= (dlt[i] - step)
        return new_times, new_sig

    def signal_regulariser(self, org_times, org_sig):
        '''
        Regularse time intervals

        @Args
        org_times (pd.Series): Original timestamps
        org_sig (pd.Series): Original signal

        @Returns:
        new_time (np.array): regularised timestamps
        new_signal (np.array): interpolated signal
        '''
        # --- Interpolate to regular time intervals
        new_times, new_sig = self._sig_reg_liner_interp(org_times, org_sig)
        # new_times, new_sig = self._sig_reg_skip_gaps(org_times, org_sig)
        # new_times, new_sig = org_times, org_sig
        return new_times, new_sig

    def signal_smoother(self, signal):
        '''
        Smoothing to reduce sensor noise by Savitzky-Golay

        @Args
        signal (pd.Series): signal

        @Returns
        smoothed (np.array): smoothed signal

        '''
        # --- Savitzky-Golay smoothing
        if self.savgol_win_len == 0 or self.savgol_polyorder == 0:
            smoothed = signal
        else:
            smoothed = savgol_filter(
                signal,
                window_length=self.savgol_win_len,
                polyorder=self.savgol_polyorder,
            )
        return smoothed

    def min_max_scale(self, src):
        '''
        min max scaling

        @Args
        src (np.array): original signal

        @Returns
        dst (np.array): min max scaled signal
        '''
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        dst = scaler.fit_transform(np.atleast_2d(src).T)[:, 0]
        return dst

    def preprocess(self, df, ch):
        '''
        Signal Preprocessing

        @Args
        df (pd.DataFrame): Original RGB signal values with timestamps
        ch (str): Target channel, 'Red', 'Green' or 'Blue'

        @Returns
        new_times (np.array): processed timestamps
        new_signal (np.array): processed signal
        '''
        # --- data
        times = df['time'].values
        signal = df[ch].values
        # --- uniform filter
        if self.uniform_filter_size != 0:
            signal = uniform_filter1d(signal, size=self.uniform_filter_size)
        # --- time interval regularising
        times, signal = self.signal_regulariser(times, signal)
        # --- smoothing
        signal = self.signal_smoother(signal)
        # --- bandpass filtering
        signal = self.bandpass_filter(signal)
        # --- min max scaling
        # signal = self.min_max_scale(signal)
        return times, signal

    def find_peaks(self, signal):
        '''
        '''
        if self.peak_distance_factor == 0:
            distance = None
        else:
            distance = self.fs * self.peak_distance_factor
        if self.peak_prominence_factor == 0:
            prominence = None
        else:
            prominence = np.std(signal) * self.peak_prominence_factor
        if self.peak_height_factor == 0:
            height = None
        else:
            sig_max = signal.max()
            sig_min = signal.min()
            height = self.peak_height_factor * (sig_max - sig_min) + sig_min
        peaks, _ = find_peaks(
            signal,
            distance=distance,
            prominence=prominence,
            height=height,
        )

        return peaks

    def metrics_extractor(self, times, signal):
        '''
        Compute cardiac metrics from a preprocessed signal.

        @Args
        times (np.array): Regularized time vector
        signal (np.array): Preprocessed signal

        @Returns
        metrics (dict): Computed metrics, {HR, rMSSD, SDNN, pNN50}
        '''
        # --- peaks
        peaks = self.find_peaks(signal)
        peak_times = times[peaks]
        # --- Time interval between consecutive peaks (milliseconds)
        ibi = np.diff(peak_times) * 1000
        # --- delta IBI
        diff_ibi = np.diff(ibi)
        # --- Heart Rate (HR)
        hr = len(peaks) * 60 / (times[-1] - times[0])
        # --- root Mean Square of Successive Differences (rMSSD)
        rmssd = np.sqrt(np.mean(diff_ibi**2))
        # --- Standard Deviation of inter-beat-intervals SDNN
        sdnn = np.std(ibi, ddof=1)
        # --- percentage of successive differences higher than 50ms
        pnn50 = np.sum(np.abs(diff_ibi) > 50) / len(diff_ibi) * 100
        # --- return dict
        metrics = {
            'HR': hr,
            'rMSSD': rmssd,
            'SDNN': sdnn,
            'pNN50': pnn50,
        }
        return metrics

    def process(self, df, ch):
        '''
        Preprocess and Metric extraction from original signal

        @Args
        df (pd.DataFrame): Original RGB signal values with timestamps
        ch (str): Target channel, 'Red', 'Green' or 'Blue'

        @Returns
        metrics (dict): Computed metrics
                        HR, rMSSD, SDNN, pNN50, and number of peaks
        '''
        timestamps, signal = self.preprocess(df, ch)
        metrics = self.metrics_extractor(timestamps, signal)
        return metrics


def main():
    import sys
    cdir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(cdir, '..'))
    sys.path.append(root_dir)
    from utils.data_loader import iter_data_df
    from utils.data_loader import expected_results_df
    processor = DataProcessor()
    for ch in processor.channels:
        print('-' * 3, ch, '-' * 10)
        all_results = {}
        for df, file_name in iter_data_df():
            results = processor.process(df, ch)
            all_results[file_name] = results
        results_df = pd.DataFrame(all_results).T
        print(results_df)
    print('-' * 3, 'expected', '-' * 10)
    print(expected_results_df())


if __name__ == '__main__':
    main()
