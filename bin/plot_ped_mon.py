#!/usr/bin/env python3

import h5py
import numpy as np
import os
import time
import fire

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from glob import glob
from tqdm.auto import tqdm


def main(indir, title='SLACube Ped. Monitor', progress=False):
    mpl.use('agg')
    sns.set_theme('talk', 'ticks')

    files = glob(os.path.join(indir, '*_ped_mon.h5'))

    ts = []
    adcs = []
    cnts = []

    for fpath in tqdm(files, disable=not progress):
        with h5py.File(fpath, 'r') as f:
            t = f['timestamp'][:]
            ped_hist = f['pedestal'][:]
			
        for i in range(len(t)):
            sel = ped_hist[i] > 0
            adcs.append(np.where(sel)[0])
            cnts.append(ped_hist[i,sel])
            ts.append(np.full(np.count_nonzero(sel), t[i]))
			
    ts = np.concatenate(ts)
    adcs = np.concatenate(adcs)
    cnts = np.concatenate(cnts)

    dates = np.vectorize(datetime.fromtimestamp)(ts)

    #!FIXME(2022-02-10 kvt) fix datetime offset	
    datenums = mpl.dates.date2num(dates) - 7*60*60 / 86400

    t_min = ts.min()
    t_max = ts.max()
    t_bins = int(np.ceil((t_max - t_min)/60))


    fig, ax = plt.subplots(figsize=(12,6))

    img = ax.hexbin(
        datenums, adcs, C=cnts, 
        gridsize=(t_bins, 256), 
        extent=(datenums.min(), datenums.max(), 0, 256), 
        cmap='viridis', bins='log'
    )

    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))

    ax.set_title(title)
    ax.set_ylabel('Mean Ped. [ADC]')
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Num. of Channels')

    fig.tight_layout()

    timestamp = hex(int(time.time()))[2:]
    fig.savefig(f'ped_mon_{timestamp}.png')

if __name__ == '__main__':
    fire.Fire(main)
