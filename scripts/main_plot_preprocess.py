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
        self.raw_data = {}
        self.metrics = {}
        self.expt_metrics = None
        self.metric_strs = {}
        self.filt_times = {}
        self.filt_data = {}
        self.filt_peaks = {}
        self.raw_peaks = {}
        self.is_filt_time_gaps = {}

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
        self.raw_data = {}
        self.metrics = {}
        self.expt_metrics = None
        self.metric_strs = {}
        self.filt_times = {}
        self.filt_data = {}
        self.filt_peaks = {}
        self.is_filt_time_gaps = {}
        # --- raw data
        self.raw_time = np.copy(df['time'].values)
        # --- mask to find time gap in raw data
        raw_time_dlt = np.diff(self.raw_time, prepend=0)
        self.is_raw_time_gap = raw_time_dlt > self.time_delta_th
        # --- Create plotting data for each channel
        for ch in self.channels:
            # --- raw data
            self.raw_data[ch] = np.copy(df[ch].values)
            # --- preprocessing
            filt_time, filt_data = self.preprocess(self.raw_time,
                                                   self.raw_data[ch])
            # --- processed data
            self.filt_data[ch] = filt_data
            # --- timestamps for processed data
            self.filt_times[ch] = filt_time
            # --- peaks in raw data
            self.raw_peaks[ch] = self.find_peaks(df[ch])
            # --- peaks in processed data
            self.filt_peaks[ch] = self.find_peaks(filt_data)
            # --- compute cardiac metrics
            self.metrics[ch] = self.metrics_extractor(filt_time, filt_data)
            filt_time_dlt = np.diff(filt_time, prepend=0)
            self.is_filt_time_gaps[ch] = filt_time_dlt > self.time_delta_th
        # --- expected metrics
        mask = self.expt_metrics_df['File'] == file_name
        self.expt_metrics = self.expt_metrics_df[mask].to_dict('list')
        self.expt_metrics = {k: v[0] for k, v in self.expt_metrics.items()}
        # --- metric texts
        self.metric_strs = {}
        expt = ','.join([
            '%s:%6.2f' % (mn, self.expt_metrics[mn])
            for mn in self.metric_names
        ])
        for ch in self.channels:
            estm = ','.join([
                '%s:%6.2f' % (mn, self.metrics[ch][mn])
                for mn in self.metric_names
            ])
            txt = '\n'.join(['Truth:%s' % expt, 'Estim:%s' % estm])
            self.metric_strs[ch] = txt

    def plot_time_v_ch_vals(self):
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

    def _plot_raw_data(self, ax, ch):
        '''
        Plots raw signals
        This method is called in plot_time_v_ch_vals.

        ax: matplotlib Axes object
        ch (str): channel ('Red', 'Green', or 'Blue')
        '''
        data_name = 'Raw'
        xpos = self.raw_time
        ypos = self.raw_data[ch]
        is_gap = self.is_raw_time_gap
        peaks = self.raw_peaks[ch]
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
        xpos = self.filt_times[ch]
        ypos = self.filt_data[ch]
        peaks = self.filt_peaks[ch]
        is_gap = self.is_filt_time_gaps[ch]
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
            txt = self.metric_strs[ch]
            prm = dict(va='top', ha='left', fontsize='small')
            ax.text(0, ypos.max(), txt, **prm)
        ax.set_title('Filtered Signal (%s)' % ch, fontsize='small')
        ax.set_xlabel('Time (s)', fontsize='small')
        ax.set_ylabel('Amplitude', fontsize='small')
        ax.tick_params(axis='both', labelsize='small')
        ax.legend(loc=8, fontsize='small', ncol=3)
        ax.grid(True)

    def savefig(self, res_dir):
        '''
        save current figure.

        @Args
        res_dir: directory path to store image
        '''
        fig_name = '-'.join(['raw_vs_filtered', self.file_name]) + '.png'
        fig_path = os.path.join(res_dir, fig_name)
        plt.savefig(fig_path)
        print('Saved: %s' % fig_path)


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
        plotter.plot_time_v_ch_vals()
        # --- save image
        plotter.savefig(res_dir)


if __name__ == '__main__':
    main()
