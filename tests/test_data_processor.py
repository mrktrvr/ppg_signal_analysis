import os
import sys
import unittest
import json
import pandas as pd

CDIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CDIR, '..'))
sys.path.insert(0, ROOT_DIR)

from utils.data_loader import iter_data_df
from libs.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        pass

    def test_metrics(self):
        processor = DataProcessor(
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
        fpath = os.path.join(CDIR, 'data', 'computed_metrics.json')
        with open(fpath, 'r') as fpr:
            ans_metrics = json.load(fpr)
        for df, file_name in iter_data_df():
            for ch in processor.channels:
                res_metrics = processor.process(df, ch)
                self.assertEqual(ans_metrics[file_name][ch], res_metrics)


if __name__ == '__main__':
    unittest.main()
