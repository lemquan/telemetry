import numpy as np
import pandas as pd
import sys
from datetime import datetime

'''
    https://cto-github.cisco.com/gist/paulduda/f72467e5b890128a4851
    look at pauls for debuggin
'''
if sys.platform == 'darwin':
    import matplotlib as mil
    mil.use('TkAgg')
    import matplotlib.pyplot as plt
    import seaborn as sea
    print "Running OS X"
elif sys.platform == 'linux' or sys.platform == 'linux2':
    print "Running Linux. No plotting available."


def convert_time_epoch(ts):

    # convert into a datetime obj with microsecs
    dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')

    # change to epoch time
    ep = ((dt-datetime(1970,1,1)).total_seconds())

    return ep

def epoch_to_time(ep):
    # to check if time is coverted correctly
    ts = datetime.utcfromtimestamp(ep)
    s = datetime.strftime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
    return s

df = pd.read_csv("data/results.csv")

# rename the columns
df.columns = ['time', 'hostname', 'delta_bytes_sent', 'delta_bytes_rec',\
                                        'gen_bytes_sent', 'gen_bytes_rec', 'interface_name']
df['status'] = 1




