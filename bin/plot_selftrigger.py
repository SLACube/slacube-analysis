#!/usr/bin/env python

import os
import fire
import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from slacube.utils import filter_data_packets, get_pkts_livetime
from slacube.ped import analyze_pedestal
from slacube.geom import load_layout_np

def make_plot(ped, rate, ch_ids, outfile, title):
    mpl.use('agg')
    sns.set_theme('talk', 'white')

    pix_loc = load_layout_np()

    fig, axes = plt.subplots(3, 1, figsize=(8, 18), sharex=True, sharey=True)

    kwargs = dict(marker='o', s=15, cmap='viridis')

    mask = ped['active']
    x, y = pix_loc[mask].T

    ax = axes[0]
    sc = ax.scatter(x, y, c=ped[mask]['mean'], **kwargs)
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='Mean [ADC]')

    ax = axes[1]
    sc = ax.scatter(
            x, y, c=ped[mask]['std'],
            **kwargs
    )
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='Std [ADC]')

    x, y = pix_loc[ch_ids].T
    ax = axes[2]
    mask = rate > 0
    sc = ax.scatter(
            x[mask], y[mask], c=rate[mask], 
            norm=mpl.colors.LogNorm(vmin=0.1, vmax=5e3),
            **kwargs
    )
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='Rate [Hz]')

    for ax in axes:
        ax.set_ylabel('y [mm]')
        axes[-1].set_xlabel('x [mm]')

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(outfile)

def save_csv(ped, rate, ch_ids, outpath):
    mask = rate > 0
    uid = ch_ids[mask]
    chip, ch = np.divmod(uid, 64)
    chip += 11

    dtype = [
        ('chip', int), ('ch', int), 
        ('mean', float), ('std', float), ('rate', float)
    ]
    arr = np.empty(len(uid), dtype=dtype)
    arr['chip'] = chip
    arr['ch'] = ch
    arr['mean'] = ped['mean'][uid]
    arr['std'] = ped['std'][uid]
    arr['rate'] = rate[mask]

    df = pd.DataFrame(arr)
    df.to_csv(outpath, index=False)

def main(fpath, outdir='./', save='png', subtitle=None, progress=False):

    print('Processing', fpath)
    with h5py.File(fpath, 'r') as f:
        pkts = f['packets']
        lifetime = get_pkts_livetime(pkts)
        mask, uids = filter_data_packets(pkts, return_uids=True)
        ch_ids, cnts = np.unique(uids, return_counts=True)
        rate = cnts / lifetime
        ped = analyze_pedestal(pkts, trunc=False, show_progress=progress)

    ofname = os.path.splitext(os.path.basename(fpath))[0] + '__selftrig'
    outpath = os.path.join(outdir, ofname)
    title = os.path.basename(fpath).split('__')[0]
    if subtitle is not None:
        title = f'{title}\n{subtitle}'

    if isinstance(save, str):
        save = (save,)
    for ftype in save:
        if ftype == 'png':
            make_plot(ped, rate, ch_ids, outpath+'.png', title)
        elif ftype == 'csv':
            save_csv(ped, rate, ch_ids, outpath+'.csv')
        else:
            print('Unsupported file format', ftype)

    print('DONE')

if __name__ == '__main__':
    fire.Fire(main)
