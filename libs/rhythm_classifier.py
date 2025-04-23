import os
import sys
import numpy as np

CDIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(ROOT_DIR)
from libs.data_processor import DataProcessor


class RuleBasedRhythmClassifier:
    '''
    A simple rule-based classifier
    to determine heartbeat regularity based on pNN50 and rMSSD values.

    '''

    def __init__(self, rmssd_th=100.0, pnn50_th=50.0):
        '''
        @Args:
        rmssd_th (float): Threshold for the rMSSD metric
                          above which rhythm is considered irregular.
        pnn50_th (float): Threshold for the pNN50 metric
                          above which rhythm is considered irregular.
        '''
        self.rmssd_th = rmssd_th
        self.pnn50_th = pnn50_th

    def is_irregular(self, rmssd, pnn50):
        '''
        Returns True if rhythm is determined as 'Irregular'
        if either pNN50 or rMSSD exceeds their respective thresholds.

        @Args
        rmssd (float): The rMSSD value computed from IBIs.
        pnn50 (float): The pNN50 value computed from IBIs.

        @Returns:
        res (Bool): True if thresholds are exceeded, otherwise False.
        '''
        condition = any([
            rmssd > self.rmssd_th,
            pnn50 > self.pnn50_th,
        ])
        res = True if condition else False
        return res


class RhythmClassifier():
    '''
    A Classifier to determine heartbeat regularity
    based on pNN50 and rMSSD values.
    '''

    def __init__(self, rmssd_th=100.0, pnn50_th=50.0, **args):
        '''
        @Args:
        rmssd_th (float): Threshold for the rMSSD metric
                          above which rhythm is considered irregular.
        pnn50_th (float): Threshold for the pNN50 metric
                          above which rhythm is considered irregular.
        '''
        self.clf = RuleBasedRhythmClassifier(rmssd_th, pnn50_th)
        self.data_processor = DataProcessor(**args)

    def _metrics(self, df, chs):
        '''
        Computes metrics for channels

        @Args
        df (pd.DataFrame): Original RGB signal values with timestamps
        chs (lsit of str): Target channels

        @Returns
        metrics (dict): Computed metrics, {ch: {HR, rMSSD, SDNN, pNN50}}
        '''
        metrics = {}
        for ch in chs:
            metrics[ch] = self.data_processor.process(df, ch)
        return metrics

    def _predict(self, metrics):
        '''
        @Args
        metrics (dict): Computed metrics, {ch: {HR, rMSSD, SDNN, pNN50}}

        @Returns
        res (Bool): True if thresholds are exceeded, otherwise False.
        '''
        res_list = []
        for ch, vals in metrics.items():
            res = self.clf.is_irregular(vals['rMSSD'], vals['pNN50'])
            res_list.append(res)
        res = bool(np.bincount(res_list).argmax())
        return res

    def is_irregular(self, df, ch=''):
        '''
        Returns True if rhythm is determined as 'Irregular'
        if either pNN50 or rMSSD exceeds their respective thresholds.

        @Args
        df (pd.DataFrame): Original RGB signal values with timestamps
        ch (str): Target channel, 'Red', 'Green', 'Blue' or ''
                  if ch == '', all channels are evaluated

        @Returns:
        res (Bool): True if thresholds are exceeded, otherwise False.
        '''
        chs = self.data_processor.channels if ch == '' else [ch]
        metrics = self._metrics(df, chs)
        res = self._predict(metrics)
        return res

    def label(self, is_irregular):
        '''
        Returns label of irregular / regular rhythm

        @Args
        is_irregular (Bool): irregular(True) or regular(False)

        @Returns
        res (str): AFib(irregular=True) or Sinus(irrregular=False)
        '''
        res = 'AFib' if is_irregular else 'Sinus'
        return res


def main_rule_based_clf():
    clf = RuleBasedRhythmClassifier()
    src_list = [
        (101, 51, True),
        (100, 51, True),
        (101, 50, True),
        (100, 50, False),
    ]
    for rmssd, pnn50, ans in src_list:
        res = clf.is_irregular(rmssd, pnn50)
        print(' | '.join([
            'Is response correct?:%5s' % (ans == res),
            'truth:%5s' % ans,
            'estim:%5s' % res,
            'rMSSD:%5.1f' % rmssd,
            'pNN50:%5.1f' % pnn50,
        ]))


def main_clf():
    from utils.data_loader import iter_data_df
    clf = RhythmClassifier()
    for df, file_name in iter_data_df():
        print('-' * 3, file_name, '-' * 10)
        for ch in clf.data_processor.channels:
            res = clf.is_irregular(df, ch)
            print('%5s:%s' % (ch, res))
        res = clf.is_irregular(df)
        print('%5s:%s' % ('all', res))


if __name__ == '__main__':
    main_rule_based_clf()
    main_clf()
