import os
import datetime
import numpy as np
import pandas as pd

IGNORED_COLUMNS = [
    'BT',                              # correlated with CPU
    'IDLA',
    'SCHA',
    'SCHE',
    'ADMA',
    'ADMN',
    'QUEA',
    'QUEU',
    'NETA',
    'NETW',
    'dDTmS',
    'dDTmU',
    'AUNDO',
    'SVTM',
    'LMON',
    'LfpA',
    'DTCc',
    'DTCf',
    'TotPGA'
]


def preprocess(directory, filenames=None):
    if filenames is None:
        filenames = [
            'sddb_db_metrics_0.dat',
            'sddb_db_metrics_1.dat',
            'sddb_db_metrics_2.dat',
            'sddb_db_metrics_3.dat',
        ]
    headers = None
    with open(os.path.join(directory, filenames[0]), 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                items = line.split()
                if items[0] == 'Date':
                    headers = items
                    break
    assert headers is not None
    print(headers)

    with open(os.path.join(directory, 'sddb_new.log'), 'w') as f_out:
        f_out.write('{}\n'.format(','.join(headers)))
        for filename in filenames:
            with open(os.path.join(directory, filename), 'r') as f_in:
                for i, line in enumerate(f_in):
                    line = line.strip()
                    if line:
                        items = line.split()
                        if items[0] == 'Date':
                            continue
                        f_out.write('{}\n'.format(','.join(items)))


def generate_minute_level(directory, filename='sddb_new.log'):
    df = pd.read_csv(os.path.join(directory, filename))
    df = df.fillna(df.median())
    df = df.drop(columns=['DBN', 'DBR', 'DG', 'SP', 'HOST'])
    columns = [c for c in df.columns if c not in ['Date', 'Time']]

    bins, timestamps = {}, []
    for index, row in df.iterrows():
        t = datetime.datetime.strptime('{} {}'.format(row['Date'], row['Time']), '%d-%m-%Y %H:%M:%S')
        t = int(t.replace(second=0).timestamp())
        if t not in bins:
            bins[t] = []
            timestamps.append(t)
        bins[t].append(row[columns].values)

    data = np.array([np.mean(bins[t], axis=0) for t in timestamps])
    data = np.concatenate([np.array(timestamps).reshape((len(timestamps), 1)), data], axis=1)
    new_df = pd.DataFrame(data, columns=['timestamp'] + columns)
    new_df.to_csv(os.path.join(directory, 'sddb_minute.log'), index=False)


def load_sddb_dataset(directory, date_range):
    df = pd.read_csv(os.path.join(directory, 'sddb_minute.log'))
    df = df.fillna(df.median())

    if date_range is not None:
        a, b = date_range
        if a is None:
            a = '1970-01-01'
        if b is None:
            b = '2050-01-01'
        a = datetime.datetime.strptime('{} 00:00:00'.format(a), '%Y-%m-%d %H:%M:%S').timestamp()
        b = datetime.datetime.strptime('{} 00:00:00'.format(b), '%Y-%m-%d %H:%M:%S').timestamp()
        df = df[df['timestamp'] >= a]
        df = df[df['timestamp'] <= b]

    new_df = df.drop(columns=['timestamp'] + IGNORED_COLUMNS)
    return new_df, new_df, None


def load_sddb_dataset_10seconds(directory, date_range):
    df = pd.read_csv(os.path.join(directory, 'sddb_new.log'))
    df = df.drop(columns=['DBN', 'DBR', 'DG', 'SP', 'HOST'])
    df = df.fillna(df.median())

    timestamps = [
        datetime.datetime.strptime('{} {}'.format(d, t), '%d-%m-%Y %H:%M:%S').timestamp()
        for d, t in zip(df['Date'], df['Time'])
    ]
    df['timestamp'] = timestamps

    if date_range is not None:
        a, b = date_range
        if a is None:
            a = '1970-01-01'
        if b is None:
            b = '2050-01-01'
        a = datetime.datetime.strptime('{} 00:00:00'.format(a), '%Y-%m-%d %H:%M:%S').timestamp()
        b = datetime.datetime.strptime('{} 00:00:00'.format(b), '%Y-%m-%d %H:%M:%S').timestamp()
        df = df[df['timestamp'] >= a]
        df = df[df['timestamp'] <= b]

    new_df = df.drop(columns=['Date', 'Time', 'timestamp'] + IGNORED_COLUMNS)
    return new_df, new_df, None


def remove_constant_timeseries(train_data, test_data, test_labels, activate=False):
    if not activate:
        return train_data, test_data, test_labels

    if type(train_data) == np.ndarray:
        m = np.mean(train_data, axis=0)
        d = np.mean(np.abs(train_data - m), axis=0)
        assert len(d) == train_data.shape[1]
        indices = [i for i in range(train_data.shape[1]) if d[i] > 1e-8]
        return train_data[:, indices], test_data[:, indices] if test_data is not None else None, test_labels
    else:
        columns = train_data.columns
        m = np.mean(train_data.values, axis=0)
        d = np.mean(np.abs(train_data.values - m), axis=0)
        assert len(d) == train_data.shape[1]
        indices = [c for i, c in enumerate(columns) if d[i] > 1e-8]
        return train_data[indices], test_data[indices] if test_data is not None else None, test_labels


def load_data(base_dir, dataset="sddb_sec", remove_constants=False, **kwargs):
    if dataset in ["sddb"]:
        data = load_sddb_dataset(base_dir, **kwargs)
    elif dataset in ["sddb_sec"]:
        data = load_sddb_dataset_10seconds(base_dir, **kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    return remove_constant_timeseries(*(data + (remove_constants,)))
