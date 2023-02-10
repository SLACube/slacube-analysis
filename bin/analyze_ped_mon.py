#!/usr/bin/env python3

import os
import fire
import h5py

from slacube.ped import monitor_pedestal

def main(fpath, duration, outdir='./', progress=False):

    print('Processing', fpath)
    with h5py.File(fpath, 'r') as f:
        ped, ts = monitor_pedestal(
            f['packets'], 
            hist_mode=True, 
            duration=duration, 
            show_progress=progress
        )

    ofname = os.path.basename(fpath).split('___')[0] + '__ped_mon.h5'
    outpath = os.path.join(outdir, ofname)
    print('Saving to', outpath)

    with h5py.File(outpath, 'w') as f:
      f.create_dataset('pedestal', data=ped, compression='gzip')
      f.create_dataset('timestamp', data=ts)
    print('DONE')

if __name__ == '__main__':
    fire.Fire(main)
