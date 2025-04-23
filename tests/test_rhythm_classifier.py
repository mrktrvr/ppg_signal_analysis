import os
import sys
import unittest
import numpy as np
from itertools import product
from itertools import combinations

CDIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CDIR, '..'))
sys.path.insert(0, ROOT_DIR)

from utils.data_loader import iter_data_df
from libs.rhythm_classifier import RhythmClassifier
from libs.rhythm_classifier import RuleBasedRhythmClassifier


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        pass

    def test_rule_based_classifier(self):
        clf = RuleBasedRhythmClassifier()
        src_list = [
            (101, 51, True),
            (100, 51, True),
            (101, 50, True),
            (100, 50, False),
        ]
        for rmssd, pnn50, ans in src_list:
            res = clf.is_irregular(rmssd, pnn50)
            self.assertEqual(res, ans)

    def test_classifier(self):
        vals = [
            (101, 51, True),
            (100, 51, True),
            (101, 50, True),
            (100, 50, False),
        ]
        clf = RhythmClassifier(
            fs=75,
            bandpass_lowcut=0.5,
            bandpass_highcut=5,
            bandpass_order=2,
            peak_distance_factor=0.5,
            peak_prominence_factor=0.5,
            uniform_filter_size=0,
            savgol_win_len=21,
            savgol_polyorder=2,
        )
        n_chs = len(clf.data_processor.channels)
        src_list = []
        for n in range(1, n_chs + 1):
            itr = product(
                combinations(clf.data_processor.channels, n),
                product(*[range(len(vals))] * n),
            )
            for cs, ix in itr:
                metrics = {
                    x1: {
                        'rMSSD': vals[x2][0],
                        'pNN50': vals[x2][1]
                    }
                    for x1, x2 in zip(cs, ix)
                }
                ans_cnd = [vals[x][2] for x in ix]
                ans = bool(np.bincount(ans_cnd).argmax())
                src_list.append((metrics, ans))
        for metrics, ans in src_list:
            res = clf._predict(metrics)
            self.assertEqual(res, ans)


if __name__ == '__main__':
    unittest.main()
