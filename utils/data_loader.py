import os
import sys
import glob
import numpy as np
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
        df = merge_channels(df)
        yield df, file_name


def merge_channels(df):
    from sklearn.decomposition import PCA
    r, g, b = df['Red'], df['Green'], df['Blue']
    # --- weighted average
    df['R2G62B'] = r * 0.2 + g * 0.6 + b * 0.2
    df['R1G1B1'] = (r + g + b) / 3
    # --- PCA
    pca = PCA(n_components=1)
    df['PCA'] = pca.fit_transform(np.vstack((r, g, b)).T).flatten()
    # --- YCbCr
    df['Y'], df['Cb'], df['Cr'] = rgb_to_ycbcr(r, g, b)
    return df


def rgb_to_ycbcr(r, g, b):
    """
    Convert RGB channels to YCbCr and return only Y (luminance) channel.

    Args:
        r, g, b (np.array): Red, Green, Blue channels.
    Returns:
        np.array: Luminance (Y) signal.
    """
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def rgb_to_hsv_v(r, g, b):
    """
    Convert RGB channels to HSV and return only V (Value) channel.

    Args:
        r, g, b (np.array): Red, Green, Blue channels.
    Returns:
        np.array: V (brightness) signal.
    """
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    v = cmax  # In HSV, Value = max(R, G, B)

    return v


def rgb_to_lab(r, g, b):
    """
    Convert RGB channels to CIE LAB and return only L (Lightness) channel.
    Approximate version without full color calibration.

    Args:
        r, g, b (np.array): Red, Green, Blue channels.

    Returns:
        np.array: L (lightness) signal.
    """
    # Step 1: RGB to XYZ
    # r = r / 255.0
    # g = g / 255.0
    # b = b / 255.0

    # Gamma correction (sRGB)
    r = np.where(r > 0.04045, ((r + 0.055) / 1.055)**2.4, r / 12.92)
    g = np.where(g > 0.04045, ((g + 0.055) / 1.055)**2.4, g / 12.92)
    b = np.where(b > 0.04045, ((b + 0.055) / 1.055)**2.4, b / 12.92)

    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b

    # Normalize for D65 white point
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # XYZ to LAB
    def f(t):
        return np.where(t > 0.008856, t**(1 / 3), (7.787 * t) + (16 / 116))

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = (116 * fy) - 16  # Only L channel

    return l


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
