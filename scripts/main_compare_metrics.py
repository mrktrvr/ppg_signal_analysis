import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

CDIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CDIR, '..'))
sys.path.append(ROOT_DIR)
from utils.data_loader import iter_data_df
from utils.data_loader import expected_results_df
from libs.data_processor import DataProcessor


def _plot_avg_std(avg_df, std_df, title, fig_path):
    '''
    Plot averages with error bars (standard deviation)

    avg_df (pd.DataFrame): average values
    std_df (pd.DataFrame): standard variation to show error bar
    title (str): title of the chart
    fig_path (str): path to save figure
    '''
    # text_prm = dict(va='bottom', ha='center', rotation=90, fontsize='small')
    n_chs = avg_df.shape[1]
    margin = 0.05
    w = (1 - 2 * margin) / n_chs
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    for i, (ch, vals) in enumerate(avg_df.items()):
        shift = (w / 2.0 + w * i)
        ypos = vals.values
        xpos = np.arange(len(ypos)) + shift + margin
        color = ch.lower()
        # ax.bar(xpos, ypos, color=color, width=w, label=ch)
        # ax.scatter(xpos, ypos, color=color, marker='+', label=ch)
        ax.errorbar(xpos, ypos, yerr=std_df[ch], color=color, fmt='x')
        # for x, y in zip(xpos, ypos):
        #     ax.text(x, y, '%.1f' % y, **text_prm)
    ax.set_xticks(np.arange(len(avg_df)) + 0.5)
    ax.set_xticklabels(avg_df.index)
    ax.tick_params(axis='x', labelsize='small')
    ax.tick_params(axis='y', labelsize='small')
    # ax.legend(loc=0, fontsize='x-small')
    ax.grid(True)
    ax.set_xlabel('Metric', fontsize='x-small')
    ax.set_ylabel('Delta', fontsize='x-small')
    ax.set_title(title, fontsize='small')
    plt.tight_layout()
    if fig_path == '':
        plt.show()
    else:
        plt.savefig(fig_path)
        print('Saved: %s' % fig_path)
        plt.close()


def _plot_multi_bar(dfs, metric_names, suptitle, fig_path=''):
    '''
    plot multi-bar chart

    @Args
    dfs (dict of pd.DataFrame): {channel: df of metrics for each signal}
    metric names (list of str): metric names
    suptitle (str): title text
    fig_path: path to save figure
    '''
    n_chs = len(dfs)
    margin = 0.05
    w = (1 - 2 * margin) / n_chs
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, metric in zip(axes.flat, metric_names):
        for i, (ch, cur_df) in enumerate(dfs.items()):
            shift = (w / 2.0 + w * i)
            ypos = cur_df[metric]
            xpos = np.arange(len(ypos)) + shift + margin
            color = ch.lower()
            ax.bar(xpos, ypos, color=color, width=w, label=ch)
            ax.plot(xpos, ypos, '_', color=color)
            ax.set_title(metric, fontsize='x-small')
        ax.set_xlabel('File', fontsize='x-small')
        ax.set_ylabel('Delta', fontsize='x-small')
        ax.set_xticks(np.arange(len(ypos)) + 0.5)
        ax.set_xticklabels([x.replace(' ', '\n') for x in cur_df.index])
        ax.tick_params(axis='x', rotation=90, labelsize='x-small')
        ax.tick_params(axis='y', labelsize='x-small')
        ax.legend(loc=0, fontsize='x-small')
        ax.grid(True)
    plt.suptitle(suptitle, fontsize='small', y=0.96)
    plt.tight_layout()
    if fig_path == '':
        plt.show()
    else:
        plt.savefig(fig_path)
        print('Saved: %s' % fig_path)
        plt.close()


class MetricsComparisonData(DataProcessor):
    '''
    Computes cardiac metrics and compare with expected metrics
    '''

    def __init__(self):
        '''
        '''
        DataProcessor.__init__(self)
        self.expt_ch = 'Expected'
        self.splitter = '.'
        self.key_col = 'File'
        self.computed_metrics_df = None
        self.expected_metrics_df = None
        self.all_metrics_df = None
        self.compared_df = None
        self.batch_process()

    def batch_process(self):
        '''
        Data preparation
        computed_metrics_df: Computed metrics for all data and chs
        expected_metrics_df: Expected result
        all_metrics_df: computed_metrics_df + expected_metrics_df
        compared_df: differences between computed and expected metrics
        '''
        self.computed_metrics_df = self._processed_metrics()
        self.expected_metrics_df = self._expected_metrics()
        self.all_metrics_df = self._merge_dfs(
            self.computed_metrics_df,
            self.expected_metrics_df,
        )
        self.compared_df = self._comparison()

    def _processed_metrics(self):
        '''
        Loads and processes PPG data and compute metrics
        '''
        # --- processed metrics
        processed_metrics = {}
        for df, file_name in iter_data_df():
            processed_metrics[file_name] = {}
            for ch in self.channels:
                metrics = self.process(df, ch)
                metrics = {'%s.%s' % (ch, k): v for k, v in metrics.items()}
                processed_metrics[file_name].update(metrics)
        df = pd.DataFrame.from_dict(processed_metrics, orient='index')
        return df

    def _expected_metrics(self):
        '''
        Loads expected metrics
        '''
        # --- expected metrics
        df = expected_results_df()
        df.columns = [
            self.splitter.join([self.expt_ch, x])
            if x in self.metric_names else x for x in df.columns
        ]
        df.index = df[self.key_col]
        df = df.drop([self.key_col], axis=1)
        return df

    def _merge_dfs(self, df1, df2):
        '''
        merge df1 and df2
        '''
        # --- all metrics by merging
        df = pd.merge(df1, df2, left_index=True, right_index=True)
        # --- column re-order
        columns = [
            self.splitter.join([c, mn]) for mn in self.metric_names
            for c in self.channels + [self.expt_ch]
        ]
        df = df[columns]
        return df

    def _comparison(self):
        '''
        Calculates differences between computed and expected metrics
        '''
        df_list = []
        for mn in self.metric_names:
            df1 = self.df_selecter(mn, self.computed_metrics_df)
            df2 = self.df_selecter(mn, self.expected_metrics_df)
            vals = df1.values - df2.values
            cols = df1.columns + self.splitter + 'diff'
            index = df1.index
            diff_df = pd.DataFrame(vals, columns=cols, index=index)
            df_list.append(diff_df)
        df = pd.concat(df_list, axis=1)
        return df


class MetricsComparison(MetricsComparisonData):
    '''
    Computes cardiac metrics and compare with the expected result.
    '''

    def __init__(self):
        '''
        '''
        MetricsComparisonData.__init__(self)

    def df_selecter(self, tar_str, df=None):
        '''
        Returns DataFrame that has columns with tar_str

        @Args
        tar_str (str): target string to select columns

        @Returns
        ret (pd.DataFrame): DataFrame with selected columns
        '''
        if df is None:
            df = self.all_metrics_df
        mask = df.columns.str.find(tar_str) != -1
        cols = list(df.columns[mask])
        ret = df[cols]
        return ret

    def show_each_channel(self):
        '''
        Shows metrics for each channel
        '''
        for ch in [self.expt_ch] + self.channels:
            print('-' * 3, ch, '-' * 70)
            print(self.df_selecter(ch).to_markdown())

    def show_each_metric(self):
        '''
        Shows metrics for each metric
        '''
        for mn in self.metric_names:
            print('-' * 3, mn, '-' * 70)
            print(self.df_selecter(mn).to_markdown())

    def show_diffs(self):
        '''
        Shows tables of differences between computed and expected metrics
        '''
        for ch in self.channels:
            print(self.df_selecter(ch, self.compared_df).to_markdown())
            print(self.df_selecter(ch, self.compared_df).mean())

    def plot_metrics(self, res_dir):
        '''
        Plot metrics
        '''
        dfs = {}
        avgs = {}
        stds = {}
        for ch in self.channels:
            cur_df = self.df_selecter(ch, self.computed_metrics_df)
            cur_df.columns = [
                x.split(self.splitter)[1] for x in cur_df.columns
            ]
            cur_df_avg = cur_df.mean()
            cur_df_std = cur_df.std()
            avg_df = pd.DataFrame(cur_df_avg, columns=['Avg']).T
            cur_df = pd.concat([cur_df, avg_df], axis=0)
            dfs[ch] = cur_df
            avgs[ch] = list(cur_df_avg)
            stds[ch] = list(cur_df_std)
        # ---
        fpath_tmp = os.path.join(res_dir, 'metrics_%s.png')
        # --- 3 channels
        fig_path = fpath_tmp % '3channels'
        suptitle = 'Computed MetricsComparison'
        _plot_multi_bar(dfs, self.metric_names, suptitle, fig_path)
        # --- Each channel
        for ch in self.channels:
            fig_path = fpath_tmp % ch
            df_1ch = {ch: dfs[ch]}
            fig_path = fpath_tmp % ch
            suptitle = 'Computed Metrics'
            _plot_multi_bar(df_1ch, self.metric_names, suptitle, fig_path)
        # --- Averages
        fig_path = fpath_tmp % 'avg'
        avg_df = pd.DataFrame(avgs, index=self.metric_names)
        std_df = pd.DataFrame(stds, index=self.metric_names)
        title = 'Average Computed Metrics'
        _plot_avg_std(avg_df, std_df, title, fig_path)

    def plot_metric_diffs(self, res_dir):
        '''
        Plot metric comparison
        '''
        # --- plot data
        dfs = {}
        avgs = {}
        stds = {}
        for ch in self.channels:
            cur_df = self.df_selecter(ch, self.compared_df)
            cur_df.columns = [
                x.split(self.splitter)[1] for x in cur_df.columns
            ]
            cur_df_avg = cur_df.mean()
            cur_df_std = cur_df.std()
            avg_df = pd.DataFrame(cur_df_avg, columns=['Avg']).T
            cur_df = pd.concat([cur_df, avg_df], axis=0)
            dfs[ch] = cur_df
            avgs[ch] = list(cur_df_avg)
            stds[ch] = list(cur_df_std)
        # ---
        fpath_tmp = os.path.join(res_dir, 'metrics_delta_%s.png')
        # --- 3 channels
        fig_path = fpath_tmp % '3channels'
        suptitle = 'Diff between Computed and Expected Metrics'
        _plot_multi_bar(dfs, self.metric_names, suptitle, fig_path)
        # --- Each channel
        for ch in self.channels:
            fig_path = fpath_tmp % ch
            df_1ch = {ch: dfs[ch]}
            suptitle = 'Diff between Computed and Expected Metrics'
            fig_path = fpath_tmp % ch
            _plot_multi_bar(df_1ch, self.metric_names, suptitle, fig_path)
        # --- Averages
        fig_path = fpath_tmp % 'avg'
        avg_df = pd.DataFrame(avgs, index=self.metric_names)
        std_df = pd.DataFrame(stds, index=self.metric_names)
        title = 'Average difference between Computed and Expected Metrics'
        _plot_avg_std(avg_df, std_df, title, fig_path)


def main():
    res_dir = os.path.join(ROOT_DIR, 'res', 'metrics')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    metrics_comparison = MetricsComparison()
    # --- Show each metric
    metrics_comparison.show_each_metric()
    # --- Show metrics for each channel
    metrics_comparison.show_each_channel()
    # --- Plot metrics
    metrics_comparison.plot_metrics(res_dir)
    # --- Plot metric comparison
    metrics_comparison.plot_metric_diffs(res_dir)


if __name__ == '__main__':
    main()
