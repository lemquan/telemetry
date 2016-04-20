import pandas as pd
import numpy as np
from datetime import datetime
import sys

if sys.platform == 'darwin':
    import matplotlib as mil

    mil.use('TkAgg')
    import matplotlib.pyplot as plt
    import seaborn as sea

    print "Running OS X"
elif sys.platform == 'linux' or sys.platform == 'linux2':
    print "Running Linux. No plotting available."


def convert_time(ts):
    # convert into a datetime obj with microsecs
    dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')

    # change to epoch time
    ep = ((dt - datetime(1970, 1, 1)).total_seconds())

    # to check if time is coverted correctly
    # ts = datetime.utcfromtimestamp(ep)
    # datetime.strftime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
    return ep


def load():
    pos_df = pd.read_csv('data/pos_bursts.csv')
    pos_df['health'] = 1
    neg_df = pd.read_csv('data/traffic_shaper.csv')
    neg_df['health'] = 0
    df = pd.concat([pos_df, neg_df])

    # rename the columns
    df.columns = ['time', 'hostname', 'delta_bytes_sent', 'delta_bytes_rec', \
            'gen_bytes_sent', 'gen_bytes_rec', 'interface_name', 'health']

    # convert the time into epoch
    df['epoch_time'] = df['time'].apply(convert_time)
    df = df.sort_values('epoch_time')

    # remove all loop backs
    df = df[(df['interface_name'] != 'tunnel-te102') | \
            (df['interface_name'] != 'GigabitEthernet0/0/0/4.100')]

    # move the columns
    cols = list(df)
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    # df = df.drop('time', 1)

    # create dummy vars for the host names
    host_dummy = pd.get_dummies(df['hostname'])
    df = host_dummy.join(df)
    df = df.drop(['hostname', 'time'], 1)

    return df


def features(df):
    # perform some feature engineering....

    df.to_csv('data/final.csv')
    print 'finished processing data...'


if __name__ == '__main__':
    df = load()
    features(df)
