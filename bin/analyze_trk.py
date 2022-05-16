#!/usr/bin/env python3

import os
import fire
import h5py
import yaml
import numpy as np

from tqdm import tqdm
from slacube.calib import AdcCalib
from slacube.geom import load_layout_np
from slacube.evt import get_n_blocks, build_event
from slacube.trk import select_tracks

def main(fpath, cfg, outdir='./', progress=False):
    
    with open(cfg, 'r') as fp:
        cfg = yaml.safe_load(fp)

    adc2mV = AdcCalib.load(**cfg['calib'])
    pix_loc = load_layout_np(**cfg.get('layout', {}))

    ofname = os.path.splitext(os.path.basename(fpath))[0]
    ofname += '__tracks.h5'
    outpath = os.path.join(outdir, ofname)

    cfg_bld = cfg['event_builder']
    n_blocks = get_n_blocks(fpath, cfg_bld['buf_size'])

    builder = build_event(fpath, **cfg_bld)

    print('Input:', fpath)
    print('Output:', outpath)

    with h5py.File(outpath, 'w-') as f:
        create = lambda name, obj : f.create_dataset(
            name, (0,), dtype=obj.dtype, maxshape=(None,), compression='gzip'
        )
        
        def fill(name, ds, obj):
            if len(obj) == 0:
                return ds
            
            obj = np.concatenate(obj)
            
            if ds is None:
                ds = create(name, obj)
            n = len(obj)
            ds.resize((len(ds) +n,))
            ds[-n:] = obj
            return ds
            
        ds_summary = None
        ds_trk = None
        ds_seg = None

        gen_evt = tqdm(builder, total=n_blocks, disable=not progress)

        for events, evt_ts in gen_evt:
            tracks, segments, summary = select_tracks(
                events, evt_ts, pix_loc, adc2mV, cfg['detprop'], cfg['selection']
            )
            if len(summary) == 0:
                continue
            ds_summary = fill('summary', ds_summary, summary)
            ds_trk = fill('tracks', ds_trk, tracks)
            ds_seg = fill('segments', ds_seg, segments)

    print('DONE')

if __name__ == '__main__':
    fire.Fire(main)
