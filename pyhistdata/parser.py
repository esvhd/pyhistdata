import numpy as np
import pandas as pd
import time
import os


HEADERS = ['open', 'high', 'low', 'close', 'volume']


def show_avail_m1_pairs(source_dir):
    csv_files = (x for x in filter(
        lambda x: x.endswith('.csv.bz2'), os.listdir(source_dir)))
    ascii_files = (x for x in filter(lambda x: x.startswith('DAT_ASCII_'),
                                     csv_files))
    m1_files = (x for x in filter(lambda x: '_M1_' in x, ascii_files))

    pairs = dict()
    for f in m1_files:
        if len(f) < 24:
            continue
        fx = f[10:16]
        year = f[20:24]
        years = pairs.get(fx, set())
        years.add(year)
        pairs[fx] = years

    return pairs


def read_fx_csv(filename,
                source_dir,
                drop_dup=False,
                compression=None):
    df = pd.read_csv(os.path.join(source_dir, filename),
                     sep=';',
                     index_col=[0],
                     header=None,
                     parse_dates=[0],
                     compression=compression)
    df.columns = HEADERS
    df.index.name = 'Datetime'

    if np.isclose(df['volume'].sum(), 0):
        df.drop(labels=['volume'], axis=1, inplace=True)

    check_dup = df.index.duplicated()
    if check_dup.any():
        print('Warning: {} has duplicated index. Count={}'.format(
            filename, np.sum(check_dup)))
    if drop_dup:
        df.drop_duplicates(inplace=True)
    return df


def load_fx(pair, source_dir, verbose=False, compression='infer'):
    '''
    Load 1-minute bar data.
    '''
    csv_files = (x for x in filter(
        lambda x: x.endswith('.csv.bz2'), os.listdir(source_dir)))
    ascii_files = (x for x in filter(lambda x: x.startswith('DAT_ASCII_'),
                                     csv_files))
    m1_files = (x for x in filter(lambda x: '_M1_' in x, ascii_files))
    fx_files = (x for x in filter(lambda x: pair in x, m1_files))

    data = []
    t0 = time.process_time()
    for f in fx_files:
        df = read_fx_csv(f, source_dir=source_dir, drop_dup=False,
                         compression=compression)
        data.append(df)

    t1 = time.process_time()

    combined = pd.concat(data)
    t2 = time.process_time()

    combined.sort_index(inplace=True)
    t3 = time.process_time()

    if verbose:
        print('File Reading took: {:.3f}, per file: {:.3f} for {} files\n'
              'Concat df took: {:.3f}\n'
              'Sort index too: {:.3f}'.format(t1 - t0,
                                              (t1 - t0) / len(data),
                                              len(data),
                                              t2 - t1,
                                              t3 - t2))
    return combined


def pct_outlier_check(df, col='close', threshold=.05):
    pct = df.pct_change()
    idx = df[col].loc[pct[col].abs() > threshold]
    return df.loc[idx.index]
