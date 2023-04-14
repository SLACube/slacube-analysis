#!/usr/bin/env python

import os
import fire
import json
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from collections import defaultdict
from slacube.geom import load_layout_np

def analyze_cfg(files, vdda, cryo):
    pix_loc = load_layout_np()

    if cryo:
        trim_scale = 2.34
        offset = 465
    else:
        trim_scale = 1.45
        offset = 210

    output = defaultdict(list)
    for fpath in files:
        with open(fpath, 'r') as fp:
            cfg = json.load(fp)
            
        reg = cfg['register_values']
        
        threshold = vdda / 256. * reg['threshold_global']
        threshold += offset
        threshold += np.array(reg['pixel_trim_dac'], dtype=float) * trim_scale
        
        uids = (reg['chip_id'] - 11 << 6) + np.arange(64)
        loc = pix_loc[uids]
        mask = ~np.any(np.isnan(loc), axis=-1)
        
        ch_mask = np.asarray(reg['channel_mask'])
        
        output['xy'].append(loc[mask])
        output['threshold'].append(threshold[mask])
        output['uid'].append(uids[mask])
        output['ch_mask'].append(ch_mask[mask])
        
    output = { k : np.concatenate(v) for k,v in output.items() } 
    return output

def make_plot(output, outfile, title):
    mpl.use('agg')
    sns.set_theme('talk', 'white')

    mask = output['ch_mask'] == 0
    x, y = output['xy'][mask].T
    value = output['threshold'][mask]
    #value = output['ch_mask']

    fig, axes = plt.subplots(1,2, figsize=(12,5))

    ax = axes[0]
    sc = ax.scatter(x, y, c=value, marker='o', s=4, cmap='viridis')

    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(sc, ax=ax, label='Threshold [mV]')

    ax = axes[1]
    ax.hist(
            value, bins=20, 
            range=(value.min(), value.max()), 
            histtype='stepfilled'
    )
    ax.set_ylabel('Counts')
    ax.set_xlabel('Threshold [mV]')
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(outfile)

def main(
        cfg_dir, vdda, outdir='./', save='png', 
        cryo=False, title=None, progress=False
):
    print('Analyzing thresholds from', cfg_dir)

    files = glob(os.path.join(cfg_dir, 'config-*.json'))
    if len(files) == 0:
        raise FileNotFoundError('No config file found', cfg_dir)

    output = analyze_cfg(files, vdda, cryo)

    label = os.path.basename(cfg_dir)
    if title is None:
        title = label

    for ftype in save.lower().split(','):
        if ftype == 'png':
            outfile = os.path.join(outdir, f'{label}.png')
            make_plot(output, outfile, title)
            print(f'Save output {outfile}')
        else:
            print('Unsupported file format', ftype)
    print('DONE')
  
if __name__ == '__main__':
    fire.Fire(main)
