#!/usr/bin/env python3

import os
import fire
import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from slacube.ped import analyze_pedestal 
from slacube.geom import load_layout_np

def make_plot(ped, outfile, title):
    mpl.use('agg')
    sns.set_theme('talk', 'white')

    pix_loc = load_layout_np()

    fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True)

    kwargs = dict(marker='o', s=15, cmap='viridis')

    mask = ped['active']
    x, y = pix_loc[mask].T

    ax = axes[0]
    sc = ax.scatter(x, y, c=ped[mask]['mean'], **kwargs)
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='Mean [ADC]')

    ax = axes[1]
    sc = ax.scatter(x, y, c=ped[mask]['std'], **kwargs)
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='Std [ADC]')

    for ax in axes:
        ax.set_ylabel('y [mm]')
        axes[-1].set_xlabel('x [mm]')

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(outfile)

def save_hdf(ped, outpath):
    print('Saving to', outpath)
    with h5py.File(outpath, 'w') as f:
        f.create_dataset('pedestal', data=ped, compression='gzip')

def save_csv(ped, outpath):
    mask = ped['active']
    uid = np.where(mask)[0]
    chip, ch = np.divmod(uid, 64)
    chip += 11

    dtype=[('chip', int), ('ch', int), ('mean', float), ('std', float)]
    arr = np.empty(len(uid), dtype=dtype)
    arr['chip'] = chip
    arr['ch'] = ch
    arr['mean'] = ped[mask]['mean']
    arr['std'] = ped[mask]['std']

    df = pd.DataFrame(arr)
    df.to_csv(outpath, index=False)

def main(fpath, outdir='./', save=('hdf','png'), progress=False):

    print('Processing', fpath)
    with h5py.File(fpath, 'r') as f:
        ped = analyze_pedestal(f['packets'], show_progress=progress)

    ofname = os.path.basename(fpath).split('___')[0] + '__ped'
    outpath = os.path.join(outdir, ofname)
    title = os.path.basename(fpath).split('__')[0]

    for ftype in save:
        if ftype == 'hdf':
            save_hdf(ped, outpath+'.h5')
        elif ftype == 'csv':
            save_csv(ped, outpath+'.csv')
        elif ftype == 'png':
            make_plot(ped, outpath+'.png', title)
        else:
            print('Unsupported file format', ftype)

    print('DONE')

if __name__ == '__main__':
    fire.Fire(main)
