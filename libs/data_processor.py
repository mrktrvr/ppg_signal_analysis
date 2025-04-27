import os

import numpy as np
import pandas as pd

from scipy.signal import butter
from scipy.signal import buttord
from scipy.signal import filtfilt
from scipy.signal import find_peaks

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


class PreProcessor():
    '''
    Data preprocessing

    preprocessor = PreProcessor()
    timestamps, signal = preprocessor.preprocess(df, ch)
    '''

    def __init__(self, **args):
        '''
        @Args
        fs (float): sampling frequency, Default 75, nyq = 0.5 * fs
        bandpass_lowcut (float): lowcut of bandpass, nyq * lowcut
        bandpass_highcut (float): highcut of bandpass, nyq * highcut
        bandpass_order (int): order ob bandpass filter
        uniform_filter_size (int): size of uniform_filter1d
        savgol_win_len (float): savgol filter window length
        savgol_polyorder (float): savgol filger polyorder
        '''
        self.fs = args.get('fs', 75)
        self.bandpass_lowcut = args.get('bandpass_lowcut', 0.5)
        self.bandpass_highcut = args.get('bandpass_highcut', 5)
        self.bandpass_order = args.get('bandpass_order', 2)
        self.uniform_filter_size = args.get('uniform_filter_size', 0)
        self.savgol_win_len = args.get('savgol_win_len', 21)
        self.savgol_polyorder = args.get('savgol_polyorder', 2)
        # self.time_gap_th = args.get('time_gap_threshold', 10 / self.fs)
        self.time_gap_th = args.get('time_gap_threshold', 0)

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
        # --- Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')
        # --- Apply the filter with zero-phase distortion
        ret = filtfilt(b, a, signal)
        return ret

    def _sig_reg_linear_interp(self, org_times, org_sig):
        '''
        Interpolate signal to regularise time intervals
        using linear interpolation

        @Args
        org_times (pd.Series): Original timestamps
        org_signal (pd.Series): Original signal

        @Returns:
        new_time (np.array): regularised timestamps
        new_signal (np.array): interpolated signal
        '''
        step = 1 / self.fs
        new_times = np.arange(org_times[0], org_times[-1], step)
        prms = dict(kind='linear', fill_value='extrapolate')
        interp_func = interp1d(org_times, org_sig, **prms)
        new_signal = interp_func(new_times)
        return new_times, new_signal

    def _sig_reg_skip_gaps(self, org_times, org_sig):
        '''
        Regularise time intervals by modifying timestamps

        @Args
        org_times (pd.Series): Original timestamps
        org_sig (pd.Series): Original signal

        @Returns:
        new_time (np.array): regularised timestamps
        new_signal (np.array): interpolated signal
        '''
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
        new_times, new_sig = self._sig_reg_linear_interp(org_times, org_sig)
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

    def z_score_normalise(self, src_signal):
        '''
        Z-Score normalisation

        @Args
        src_signal (np.array): original signal

        @Returns
        dst_signal (np.array): processed signal
        '''
        mean = np.mean(src_signal)
        std = np.std(src_signal)
        dst_signal = (src_signal - mean) / std
        return dst_signal

    def z_score_normalise_sliding_window(self, src_signal, win_size):
        half_window = win_size // 2
        dst_signal = np.zeros_like(src_signal)
        for i in range(len(src_signal)):
            start = max(0, i - half_window)
            end = min(len(src_signal), i + half_window)
            window = src_signal[start:end]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                dst_signal[i] = (src_signal[i] - mean) / std
            else:
                dst_signal[i] = 0
        return dst_signal

    def time_gap_filler(self, times, signal):
        if self.time_gap_th == 0:
            return times, signal
        diffs = np.diff(times, prepend=0)
        condition = diffs > self.time_gap_th
        if sum(condition) == 0:
            new_times = times
            new_signal = signal
        else:
            new_times_list = []
            new_sigs_list = []
            large_diffs = diffs[condition]
            locs = np.where(condition)[0]
            dt_expt = 1 / self.fs
            i = 0
            for d, l in zip(large_diffs, locs):
                new_times_list.append(times[i:l])
                new_sigs_list.append(signal[i:l])
                n_missing = int(np.round(d / dt_expt)) - 1
                dt_bgn = times[l - 1]
                dt_end = times[l] - (1 / self.fs)
                filled_times = np.linspace(dt_bgn, dt_end, n_missing)
                if l > n_missing:
                    filled_sigs = signal[l - n_missing:l]
                else:
                    n_rep = n_missing // l
                    n_res = n_missing % l
                    res = signal[l - n_res:l]
                    filled_sigs = np.concat([signal[0:l]] * n_rep + [res])
                new_times_list.append(filled_times)
                new_sigs_list.append(filled_sigs)
                i = l
            new_times_list.append(times[i:])
            new_sigs_list.append(signal[i:])
            new_times = np.concatenate(new_times_list)
            new_signal = np.concatenate(new_sigs_list)
        return new_times, new_signal

    def preprocess(self, times, signal):
        '''
        Signal Preprocessing

        @Args
        times (np.array): raw timestamps
        signal (np.array): raw signal

        @Returns
        times (np.array): processed timestamps
        signal (np.array): processed signal
        '''
        # --- uniform filter
        if self.uniform_filter_size != 0:
            signal = uniform_filter1d(signal, size=self.uniform_filter_size)
        # --- time gap filling
        times, signal = self.time_gap_filler(times, signal)
        # --- time interval regularising
        times, signal = self.signal_regulariser(times, signal)
        # --- smoothing
        signal = self.signal_smoother(signal)
        # --- bandpass filtering
        signal = self.bandpass_filter(signal)
        # --- z-score normalisation
        # signal = self.z_score_normalise(signal)
        # signal = self.z_score_normalise_sliding_window(signal, 20)
        # --- min max scaling
        # signal = self.min_max_scale(signal)
        return times, signal


class MetricsExtractor():
    '''
    metric extractor from PPG signal

    metric_extractor = MetricExtractor()
    metrics = self.metrics_extractor(timestamps, signal)
    '''

    def __init__(self, **args):
        '''
        @Args
        fs (float): sampling frequency, Default 75, nyq = 0.5 * fs
        peak_distance_factor (float): factor of find_peak distance
                                      fs * peak_distance_factor
        peak_prominence_factor (float): factor of find_peak prominence
                                        np.std(signal) * peak_prominence_factor
        '''
        self.fs = args.get('fs', 75)
        self.peak_distance_factor = args.get('peak_distance_factor', 0.5)
        self.peak_prominence_factor = args.get('peak_prominence_factor', 0.5)
        self.ibi_q_low = args.get('ibi_q_low', 0)
        self.ibi_q_high = args.get('ibi_q_hight', 1)

    def find_peaks(self, signal):
        '''
        Find peaks from signal

        @Args
        signal (np.array): processed signal

        @Returns
        peaks (np.array): Indices of peaks in signal
        '''
        distance = self.fs * self.peak_distance_factor
        prominence = np.std(signal) * self.peak_prominence_factor
        peaks, _ = find_peaks(
            signal,
            distance=distance,
            prominence=prominence,
        )
        return peaks

    def peaks_and_ibis(self, times, signal):
        '''
        Find peaks and time delta between peaks

        @Args
        times (np.array): Regularized time vector
        signal (np.array): Preprocessed signal

        @Returns
        peaks (np.array): indices of peaks
        ibi (np.array): time delta between peaks
        '''
        # --- peaks
        peaks = self.find_peaks(signal)
        peak_times = times[peaks]
        # --- Time interval between consecutive peaks (milliseconds)
        ibi = np.diff(peak_times) * 1000
        if len(ibi) != 0:
            mask = np.all([
                ibi > np.quantile(ibi, q=self.ibi_q_low),
                ibi < np.quantile(ibi, q=self.ibi_q_high),
            ], 0)
            ibi = ibi[mask]
        return peaks, ibi

    def metrics(self, times, peaks, ibi):
        '''
        Compute cardiac metrics.

        @Args
        times (np.array): Regularized time vector
        peaks (np.array): indices of peaks
        ibi (np.array): time delta between peaks

        @Returns
        metrics (dict): Computed metrics, {HR, rMSSD, SDNN, pNN50}
        '''
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

    def metrics_extractor(self, times, signal):
        '''
        Compute cardiac metrics from a preprocessed signal.

        @Args
        times (np.array): Regularized time vector
        signal (np.array): Preprocessed signal

        @Returns
        metrics (dict): Computed metrics, {HR, rMSSD, SDNN, pNN50}
        '''
        peaks, ibi = self.peaks_and_ibis(times, signal)
        metrics = self.metrics(times, peaks, ibi)
        return metrics


class DataProcessor(PreProcessor, MetricsExtractor):
    '''
    '''

    def __init__(self, **args):
        '''
        @Args
        fs (float): sampling frequency, Default 75, nyq = 0.5 * fs
        bandpass_lowcut (float): lowcut of bandpass, nyq * lowcut
        bandpass_highcut (float): highcut of bandpass, nyq * highcut
        bandpass_order (int): order ob bandpass filter
        uniform_filter_size (int): size of uniform_filter1d
        savgol_win_len (float): savgol filter window length
        savgol_polyorder (float): savgol filger polyorder
        peak_distance_factor (float): factor of find_peak distance
                                      fs * peak_distance_factor
        peak_prominence_factor (float): factor of find_peak prominence
                                        np.std(signal) * peak_prominence_factor
        '''
        PreProcessor.__init__(self, **args)
        MetricsExtractor.__init__(self, **args)

        self.channels = [
            'Red',
            'Green',
            'Blue',
            # 'R1G1B1',
            # 'R2G62B',
            # 'PCA',
            'Y',
            # 'Cb',
            # 'Cr',
        ]
        self.metric_names = ['HR', 'rMSSD', 'SDNN', 'pNN50']

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
        times, signal = df['time'].values, df[ch].values
        times, signal = self.preprocess(times, signal)
        metrics = self.metrics_extractor(times, signal)
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
