import os
import sys
import glob
import pandas as pd

CDIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CDIR, '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')


def iter_data_file_names(data_dir=''):
    '''
    file name iterator

    @Args
    data_dir (str): data directory path, default DATA_DIR
    @Returns
    file_name (str): file name (basename)
    '''
    if data_dir == '':
        data_dir = DATA_DIR
    csv_files = data_file_paths(data_dir)
    for file_path in csv_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        yield file_name


def data_file_paths(data_dir=''):
    '''
    Returns list of data file paths

    @Args
    data_dir (str): data directory path, default DATA_DIR
    @Returns
    fpaths (str): list of data file paths
    '''
    if data_dir == '':
        data_dir = DATA_DIR
    fpaths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    fpaths = [f for f in fpaths if os.path.basename(f).startswith('2022')]
    fpaths = sorted(fpaths)
    return fpaths


FILE_NO_MAP = {v: '%02d' % i for i, v in enumerate(iter_data_file_names())}


def iter_data_file_paths(data_dir=''):
    '''
    data file path iterator

    @Args
    data_dir (str): data directory path, default DATA_DIR
    @Returns
    file_path (str): path to data file
    '''
    if data_dir == '':
        data_dir = DATA_DIR
    csv_files = data_file_paths(data_dir)
    for file_path in csv_files:
        yield file_path


def csv_to_df(file_path):
    try:
        df = pd.read_csv(file_path)
    except IOError as e:
        raise e
    except Exception as e:
        raise e
    return df


def iter_data_df():
    '''
    Data iterator

    @Returns
    df (pd.DataFrame): signal (time, Red, Green, Blue)
    file_name (str): file number
    '''
    csv_files = data_file_paths(DATA_DIR)
    for file_path in csv_files:
        df = csv_to_df(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_name = FILE_NO_MAP[file_name]
        yield df, file_name


def expected_results_df():
    '''
    Returns expected_results in pd.DataFrame

    @Returns
    df pd.DataFrame: expected results
    '''
    file_path = os.path.join(DATA_DIR, 'expected_results.csv')
    df = csv_to_df(file_path)
    df['File'] = [FILE_NO_MAP[x] for x in df['File']]
    return df


def main():
    print('--- data paths ---')
    print('\n'.join(data_file_paths(DATA_DIR)))
    print('-' * 100)
    for df, fname in iter_data_df():
        print(df.shape)

    for k, v in FILE_NO_MAP.items():
        print(k, v)


if __name__ == '__main__':
    main()
