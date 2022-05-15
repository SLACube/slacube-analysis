#!/usr/bin/env python3

import os
import fire
import h5py

from slacube.ped import analyze_pedestal 

def main(fpath, outdir='./', progress=False):

    print('Processing', fpath)
    with h5py.File(fpath, 'r') as f:
        output = analyze_pedestal(f['packets'], show_progress=progress)

    ofname = os.path.splitext(os.path.basename(fpath))[0] + '__ped.h5'
    outpath = os.path.join(outdir, ofname)
    print('Saving to', outpath)

    with h5py.File(outpath, 'w') as f:
      f.create_dataset('pedestal', data=output, compression='gzip')
    print('DONE')

if __name__ == '__main__':
    fire.Fire(main)
