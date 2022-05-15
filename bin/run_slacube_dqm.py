#!/usr/bin/env python3

import os
import fire
import sys
import numpy as np

from slacube.dqm import analyze_packet_rate

def dqm(fpath, tz, outdir='./'):
    if not os.path.isdir(outdir):
        print(f'{outdir} deos not exist', file=sys.stderr)
        sys.exit(1)


    print(f'Analyzing packet rate for {fpath}') 
    output = analyze_packet_rate(fpath, 60, tz)
    
    prefix = os.path.splitext(os.path.basename(fpath))[0]
    outpath = os.path.join(outdir, f'{prefix}__pkt_rate.npz')
    print(f'Saving output to {outpath}') 

    np.savez_compressed(outpath, **output)

    print('Done\n')

if __name__ == '__main__':
    fire.Fire(dqm)
