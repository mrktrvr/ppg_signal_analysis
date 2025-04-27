import os
import sys
import numpy as np
from matplotlib import pyplot as plt

CDIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(ROOT_DIR)
from utils.data_loader import iter_data_df
from utils.data_loader import expected_results_df
from libs.data_processor import DataProcessor


class Plotter(DataProcessor):
    '''
    To process the preprocessing for signals and
    create time series chart to show before and after preprocessing signals.

    # --- plotter class
    plotter = Plotter()
    # --- set current data
    plotter.set_data(df, file_name)
    # --- plot signals
    plotter.plot_time_v_ch_vals()
    # --- save image
    plotter.savefig(res_dir)
    '''

    def __init__(self, **args):
        '''
        @Args
        time_delta_th (float): Threshold to decide time gaps in singals.
                               Default 0.2
        show_metrics (bool): Flag to show computed metrics. default False
        '''
        self.time_delta_th = args.get('time_delta_th', 0.2)
        self.show_metrics = args.get('show_metrics', False)
        DataProcessor.__init__(self)
        self.expt_metrics_df = expected_results_df()
        self.df = None
        self.file_name = None
        self.signals = {}
        self.times = {}
        self.peaks = {}
        self.ibis = {}
        self.is_gaps = {}
        self.computed_metrics = {}

        self.expt_metrics = None

    def set_data(self, df, file_name):
        '''
        Sets data to plot.
        Preprocessing for signals should be done in this method.

        @Args
        df (pd.DataFrame): Original RGB signal values with timestamps
        file_name (str): File name of the data
        '''
        self.df = df
        self.file_name = file_name
        self.signals = {'raw': {}, 'processed': {}}
        self.times = {'raw': {}, 'processed': {}}
        self.peaks = {'raw': {}, 'processed': {}}
        self.ibis = {'raw': {}, 'processed': {}}
        self.is_gaps = {'raw': {}, 'processed': {}}
        self.computed_metrics = {'raw': {}, 'processed': {}}
        self.expt_metrics = None
        # --- raw data
        # --- Create plotting data for each channel
        for ch in self.channels:
            # --- raw data
            raw_signal = np.copy(df[ch].values)
            raw_times = np.copy(df['time'].values)
            # --- mask to find time gap in raw data
            raw_time_dlt = np.diff(raw_times, prepend=0)
            self.is_gaps['raw'][ch] = raw_time_dlt > self.time_delta_th
            # --- preprocessing
            filt_times, filt_signal = self.preprocess(raw_times, raw_signal)
            self.signals['raw'][ch] = raw_signal
            self.signals['processed'][ch] = filt_signal
            self.times['raw'][ch] = np.copy(raw_times)
            self.times['processed'][ch] = filt_times
            # --- peaks in raw data
            raw_peaks, raw_ibi = self.peaks_and_ibis(raw_times, raw_signal)
            self.peaks['raw'][ch] = raw_peaks
            self.ibis['raw'][ch] = raw_ibi
            # --- peaks in processed data
            filt_peaks, filt_ibi = self.peaks_and_ibis(filt_times, filt_signal)
            self.peaks['processed'][ch] = filt_peaks
            self.ibis['processed'][ch] = filt_ibi
            # --- compute cardiac metrics
            metrics = self.metrics_extractor(filt_times, filt_signal)
            self.computed_metrics['processed'][ch] = metrics
            filt_time_dlt = np.diff(filt_times, prepend=0)
            self.is_gaps['processed'][ch] = filt_time_dlt > self.time_delta_th
        # --- expected metrics
        mask = self.expt_metrics_df['File'] == file_name
        self.expt_metrics = self.expt_metrics_df[mask].to_dict('list')
        self.expt_metrics = {k: v[0] for k, v in self.expt_metrics.items()}

    def plot_peaks_and_ibi(self, res_dir):
        n_cols = 2
        n_rows = len(self.channels)
        fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
        i = 1
        n_bins = 101
        for ch in self.channels:
            ibi = self.ibis['processed'][ch]
            dlt_ibi = np.abs(np.diff(ibi))
            ibi_bins = np.linspace(500, 1500, n_bins)
            dlt_ibi_bins = np.linspace(0, 500, n_bins)
            # --- IBI
            ax = fig.add_subplot(n_rows, n_cols, i)
            ax.hist(ibi, bins=ibi_bins)
            ax.set_title('IBI %s' % ch, fontsize='small')
            ax.set_xlabel('IBI', fontsize='small')
            ax.set_ylabel('Freq', fontsize='small')
            ax.tick_params(axis='both', labelsize='small')
            ax.grid(True)
            i += 1
            # --- Delta IBI
            ax = fig.add_subplot(n_rows, n_cols, i)
            ax.hist(dlt_ibi, bins=dlt_ibi_bins)
            ax.set_title('Delta IBI %s' % ch, fontsize='small')
            ax.set_xlabel('Delta IBI', fontsize='small')
            ax.set_ylabel('Freq', fontsize='small')
            ax.tick_params(axis='both', labelsize='small')
            ax.grid(True)
            i += 1
        suptitle = self.file_name
        plt.suptitle(suptitle)
        plt.tight_layout()
        self.savefig(res_dir, 'peaks_ibis')

    def plot_time_v_ch_vals(self, res_dir):
        '''
        Plots before and after signals for each channel.
        '''
        n_cols = 2
        n_rows = len(self.channels)
        fig = plt.figure(figsize=(6 * n_cols, 3 * n_rows))
        i = 1
        for ch in self.channels:
            # --- Raw data
            ax = fig.add_subplot(n_rows, n_cols, i)
            self._plot_raw_data(ax, ch)
            i += 1
            # --- Filter data
            ax = fig.add_subplot(n_rows, n_cols, i)
            self._plot_filtered_data(ax, ch)
            i += 1
        suptitle = self.file_name
        plt.suptitle(suptitle)
        plt.tight_layout()
        self.savefig(res_dir, 'raw_vs_filtered')

    def savefig(self, res_dir, fig_name):
        '''
        save current figure.

        @Args
        res_dir: directory path to store image
        '''
        fig_name = '-'.join([fig_name, self.file_name]) + '.png'
        fig_path = os.path.join(res_dir, fig_name)
        plt.savefig(fig_path)
        print('Saved: %s' % fig_path)

    def _plot_raw_data(self, ax, ch):
        '''
        Plots raw signals
        This method is called in plot_time_v_ch_vals.

        ax: matplotlib Axes object
        ch (str): channel ('Red', 'Green', or 'Blue')
        '''
        data_name = 'Raw'
        xpos = self.times['raw'][ch]
        ypos = self.signals['raw'][ch]
        peaks = self.peaks['raw'][ch]
        is_gap = self.is_gaps['raw'][ch]
        lbl = '%s(%d)' % (data_name, len(ypos))
        col = 'cyan'
        ax.plot(xpos, ypos, '-', label=lbl, color=col)
        lbl = '%s time gap(%d)' % (data_name, sum(is_gap))
        col = 'blue'
        ax.plot(xpos[is_gap], ypos[is_gap], 'x', label=lbl, color=col)
        lbl = '%s peaks(%d)' % (data_name, len(peaks))
        col = 'black'
        ax.plot(xpos[peaks], ypos[peaks], 'x', label=lbl, color=col)
        ax.set_title('Raw Signal(%s)' % ch, fontsize='small')
        ax.set_xlabel('Time(s)', fontsize='small')
        ax.set_ylabel('Amplitude', fontsize='small')
        ax.tick_params(axis='both', labelsize='small')
        ax.legend(loc=8, fontsize='small', ncol=3)
        ax.grid(True)

    def _plot_filtered_data(self, ax, ch):
        '''
        Plots preprocessed signals
        This method is called in plot_time_v_ch_vals.

        ax: matplotlib Axes object
        ch (str): channel ('Red', 'Green', or 'Blue')
        '''
        data_name = 'Filtered'
        xpos = self.times['processed'][ch]
        ypos = self.signals['processed'][ch]
        peaks = self.peaks['processed'][ch]
        is_gap = self.is_gaps['processed'][ch]
        lbl = '%s(%d)' % (data_name, len(ypos))
        col = 'orange'
        ax.plot(xpos, ypos, label=lbl, color=col)
        lbl = '%s time gap(%d)' % (data_name, sum(is_gap))
        col = 'black'
        ax.plot(xpos[is_gap], ypos[is_gap], 'x', label=lbl, color=col)
        lbl = '%s peaks(%d)' % (data_name, len(peaks))
        col = 'red'
        ax.plot(xpos[peaks], ypos[peaks], 'x', label=lbl, color=col)
        if self.show_metrics:
            expt = ','.join([
                '%s:%6.2f' % (mn, self.expt_metrics[mn])
                for mn in self.metric_names
            ])
            estm = ','.join([
                '%s:%6.2f' % (mn, self.computed_metrics[ch][mn])
                for mn in self.metric_names
            ])
            txt = '\n'.join(['Truth:%s' % expt, 'Estim:%s' % estm])
            prm = dict(va='top', ha='left', fontsize='small')
            ax.text(0, ypos.max(), txt, **prm)
        ax.set_title('Filtered Signal (%s)' % ch, fontsize='small')
        ax.set_xlabel('Time (s)', fontsize='small')
        ax.set_ylabel('Amplitude', fontsize='small')
        ax.tick_params(axis='both', labelsize='small')
        ax.legend(loc=8, fontsize='small', ncol=3)
        ax.grid(True)


def main():
    # --- destination directory
    res_dir = os.path.join(ROOT_DIR, 'res', 'raw_vs_filtered')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # --- plotter class
    plotter = Plotter()
    # --- for each file
    for df, file_name in iter_data_df():
        # --- set current data
        plotter.set_data(df, file_name)
        # --- plot signals
        plotter.plot_time_v_ch_vals(res_dir)
        # --- plot ibis
        plotter.plot_peaks_and_ibi(res_dir)


if __name__ == '__main__':
    main()
