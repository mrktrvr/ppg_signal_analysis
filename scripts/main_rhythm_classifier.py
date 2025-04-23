import os
import sys
import numpy as np
import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(ROOT_DIR)
from libs.rhythm_classifier import RhythmClassifier
from utils.data_loader import iter_data_df
from utils.data_loader import expected_results_df


def _classifiation_by_computed_metrics(clf):
    all_res = {}
    for df, file_name in iter_data_df():
        res_dic = {}
        for ch in clf.data_processor.channels:
            ret = clf.is_irregular(df, ch)
            res = 'AFib' if ret else 'Sinus'
            res_dic[ch] = res
        ret = clf.is_irregular(df)
        res = clf.label(ret)
        res_dic['All'] = res
        all_res[file_name] = res_dic
    res_df = pd.DataFrame(all_res).T
    return res_df


def _classifiation_by_expected_metrics(clf):
    df = expected_results_df()
    df.index = df.File
    all_res = {}
    for i, row in df.iterrows():
        ret = clf._predict({'expected': dict(row)})
        res = clf.label(ret)
        all_res[i] = {'Expetcted': res}
    res_df = pd.DataFrame(all_res).T
    return res_df


def main():
    clf = RhythmClassifier()
    computed_res_df = _classifiation_by_computed_metrics(clf)
    res_dir = os.path.join(ROOT_DIR, 'res', 'classification')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    file_path = os.path.join(res_dir, 'sinus_afib.csv')
    computed_res_df.to_csv(file_path)
    print('Saved: %s' % file_path)
    print('Classification results from computed metrics')
    print(computed_res_df)
    expected_res_df = _classifiation_by_expected_metrics(clf)
    print('Classification results from expected metrics')
    print(expected_res_df)


if __name__ == '__main__':
    main()
