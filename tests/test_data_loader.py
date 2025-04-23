import os
import sys
import unittest
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from utils.data_loader import DATA_DIR
from utils.data_loader import FILE_NO_MAP
from utils.data_loader import iter_data_df
from utils.data_loader import expected_results_df


class TestDataGenerator(unittest.TestCase):

    def setUp(self):
        pass

    def test_data_files(self):
        self.assertTrue(os.path.exists(DATA_DIR))
        self.assertEqual(len(os.listdir(DATA_DIR)), 11)

    def test_file_no(self):
        ans = {
            '2022-06-07 09-15-58': '00',
            '2022-06-07 09-51-55': '01',
            '2022-06-07 10-03-36': '02',
            '2022-06-07 11-04-55': '03',
            '2022-06-07 11-22-35': '04',
            '2022-06-14 09-31-19': '05',
            '2022-06-14 09-42-01': '06',
            '2022-06-14 11-07-24': '07',
            '2022-06-14 11-55-02': '08',
            '2022-06-14 12-55-43': '09'
        }
        self.assertEqual(FILE_NO_MAP, ans)

    def test_iter_data_df(self):
        size_list = [
            (4464, 5),
            (4480, 5),
            (4496, 5),
            (4480, 5),
            (4256, 5),
            (4448, 5),
            (4496, 5),
            (4480, 5),
            (4496, 5),
            (4480, 5),
        ]
        for i, (df, fname) in enumerate(iter_data_df()):
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIsInstance(fname, str)
            self.assertEqual(fname, '%02d' % i)
            self.assertEqual(df.shape, size_list[i])
            self.assertIn('time', df)
            self.assertIn('Red', df)
            self.assertIn('Green', df)
            self.assertIn('Blue', df)

    def test_expected_results_df(self):
        df = expected_results_df()
        self.assertEqual(df.shape, (10, 5))
        self.assertIn('HR', df)
        self.assertIn('rMSSD', df)
        self.assertIn('SDNN', df)
        self.assertIn('pNN50', df)
        self.assertEqual(
            list(df['File'].values),
            ['%02d' % i for i in range(len(df))],
        )


if __name__ == '__main__':
    unittest.main()
