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
    '''
    Classifitacion using computed metrics

    @Args
    clf: Rhythm classifier object
    @Returns
    res_df (pd.DataFrame): result in DataFramw
    '''
    all_res = {}
    # --- each data file
    for df, file_name in iter_data_df():
        res_dic = {}
        # --- each channel
        for ch in clf.data_processor.channels:
            # --- classification
            ret = clf.is_irregular(df, ch)
            # --- label to the classificatoin result
            res = clf.label(ret)
            # --- store result
            res_dic[ch] = res
        # --- all channel
        # --- classification
        ret = clf.is_irregular(df)
        # --- label to the classificatoin result
        res = clf.label(ret)
        # --- store result
        res_dic['All'] = res
        # --- store all result for each file
        all_res[file_name] = res_dic
    # --- dictionary to pd.DataFrame
    res_df = pd.DataFrame(all_res).T
    return res_df


def _classifiation_by_expected_metrics(clf):
    '''
    Classifitacion using expected metrics

    @Args
    clf: Rhythm classifier object
    @Returns
    res_df (pd.DataFrame): result in DataFramw
    '''
    df = expected_results_df()
    df.index = df.File
    all_res = {}
    # --- for each row
    for i, row in df.iterrows():
        # --- classification
        ret = clf._predict({'expected': dict(row)})
        # --- label to the classificatoin result
        res = clf.label(ret)
        # --- store result
        all_res[i] = {'Expetcted': res}
    # --- dictionary to pd.DataFrame
    res_df = pd.DataFrame(all_res).T
    return res_df


def main():
    # --- directory to store results
    res_dir = os.path.join(ROOT_DIR, 'res', 'classification')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # --- classifier object
    clf = RhythmClassifier()
    # --- classifitacion using computed metrics
    computed_res_df = _classifiation_by_computed_metrics(clf)
    # --- cassifitacion using expected metrics
    expected_res_df = _classifiation_by_expected_metrics(clf)
    # --- store classifiation result in csv
    file_path = os.path.join(res_dir, 'sinus_afib.csv')
    computed_res_df.to_csv(file_path)
    print('Saved: %s' % file_path)
    # --- print result
    print('Classification results from computed metrics')
    print(computed_res_df)
    print('Classification results from expected metrics')
    print(expected_res_df)


if __name__ == '__main__':
    main()
