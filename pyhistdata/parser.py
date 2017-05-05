import numpy as np
import pandas as pd
import time
import os
import pytz
import joblib as job
import matplotlib.pyplot as plt


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


def correct_errors(df, fx_pair, error_datastore):
    '''
    Given a FX pair data, use exising HDFS5 data to drop some bad points.

    TODO: use either a dataframe or file_path for error_datastore.

    Parameters:
    df:
        data in pandas dataframe
    fx_pair:
        fx cross, eg. AUDUSD
    error_datastore:
        HDF5 store file that contains error data. Assume key is 'errors'.
    '''
    if isinstance(error_datastore, pd.DataFrame):
        errors = error_datastore
    else:
        errors = pd.read_hdf(error_datastore, key='errors', mode='a')

    errors = errors.loc[errors.fx == fx_pair]
    df_mod = df.drop(labels=errors.stamp, axis=0)
    return df_mod


def load_fx(pair, source_dir, verbose=False, compression='infer',
            tz=None, errors_df=None):
    '''
    Load 1-minute bar data.

    Parameters:
    pair:
        FX pair name
    source_dir:
        directory that contains all the CSV data files.
    tz:
        timezone adjustment
    errors_df:
        data source for error correction info.
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

    # convert timezone if needed.
    if tz is not None:
        zone = pytz.timezone(tz)
        est_tz = pytz.timezone('US/Eastern')
        est_stamps = [est_tz.localize(x) for x in combined.index]
        new_stamps = [x.astimezone(zone) for x in est_stamps]
        combined.index = new_stamps

    if errors_df:
        combined = correct_errors(combined,
                                  fx_pair=pair,
                                  error_datastore=errors_df)

    if verbose:
        print('File Reading took: {:.3f}, per file: {:.3f} for {} files\n'
              'Concat df took: {:.3f}\n'
              'Sort index took: {:.3f}'.format(t1 - t0,
                                               (t1 - t0) / len(data),
                                               len(data),
                                               t2 - t1,
                                               t3 - t2))

    return combined


def pct_outlier_check(df, col='close', threshold=.05):
    pct = df.pct_change()
    idx = df[col].loc[pct[col].abs() > threshold]
    return df.loc[idx.index]


def _check_data(x, source_dir, threshold=.05, errors_df=None):
    '''
    Helper function for using joblib in `data_check_all()`.
    '''
    fx = load_fx(x, source_dir=source_dir, errors_df=errors_df)
    res = pct_outlier_check(fx, threshold=threshold)
    return (x, res)


def data_check_all(source_dir, n_jobs=4, pct_threshold=.05, verbose=False,
                   errors_df=None):
    all_pairs = show_avail_m1_pairs(source_dir)
    data_check = job.Parallel(n_jobs=n_jobs)(job.delayed(_check_data)
                                             (x, source_dir=source_dir,
                                              threshold=pct_threshold,
                                              errors_df=errors_df)
                                             for x in all_pairs.keys())

    focus = dict()
    for x, df in data_check:
        if len(df) > 0:
            focus[x] = df
            if verbose:
                print('Found > {:.1%} jumps in data for {}, count = {}, '
                      'max datetime = {}'
                      .format(pct_threshold, x, len(df), df.index.max()))

    if verbose:
        print('total pairs = {}'.format(len(all_pairs)))
        print('No. of pairs with potential data issue = {}'
              .format(len(focus.keys())))

    return focus


def plot_subset(data, index, offset=10, col='close', title=None):
    '''
    plot a subset of data points around each index value provided.

    Parameters:
    data:
        data to plot
    index:
        set of index values to plot, a number of data points before and after
        each index are also included.
        Max allowed no. of indices is 10.
    offset:
        number of data points before and after the index value to include in
        the plot.
    col:
        data column to plot. Default is 'close'
    '''
    # first validate index are all found in data's index
    mask = index.isin(data.index)
    if not mask.all():
        raise AssertionError('index not found in data: {}'.format(index[mask]))
    n = len(index)
    if n > 10:
        print('Too many indices to plot. Max = 10')
        return
    fig, axarr = plt.subplots(nrows=n, ncols=1)
    w, h = plt.rcParams['figure.figsize']
    fig.set_size_inches((w, n * h))

    if n > 1:
        i = 0
        for ind in index:
            iloc = data.index.get_loc(ind)
            left = max(0, iloc - offset)
            right = min(len(data) - 1, iloc + offset)
            data.iloc[left:right].loc[:, col].plot(ax=axarr[i],
                                                   ls='',
                                                   marker='x')
            if title is None:
                axarr[i].set_title(ind)
            else:
                axarr[i].set_title('{}, {}'.format(title, ind))
            i += 1
        plt.tight_layout(h_pad=.4)
    else:
        # plot only 1 axes
        iloc = data.index.get_loc(index[0])
        left = max(0, iloc - offset)
        right = min(len(data) - 1, iloc + offset)
        data.iloc[left:right].loc[:, col].plot(ax=axarr, ls='', marker='x')
        if title is None:
            axarr.set_title(ind)
        else:
            axarr.set_title('{}, {}'.format(title, index[0]))
    plt.show()
    plt.close(fig)


def plot_subset_fx(pair, source_dir, index, offset=10, col='close', tz=None):
    '''
    Load fx from csv data file and then run plot_subset.
    '''
    data = load_fx(pair=pair, source_dir=source_dir, verbose=False, tz=tz)
    plot_subset(data, index, offset=offset, col=col)
