#!/usr/bin/env python3

import fire
import os
import shutil
import numpy as np

from slacube.ped import process_pedestal
from slacube.utils import Adc2Charge
from slacube.geom import load_geom
from slacube.trk import process_file

def main(data_file, pedestal_file):
    pix_loc = load_geom()

    ped = np.load(pedestal_file)
    adc2charge = Adc2Charge(ped, vdda=1860)

    tracks, segments, summary = process_file(data_file, pix_loc, adc2charge)

    tracks_toc = np.array(list(map(len, tracks)))
    tracks_toc = np.insert(tracks_toc.cumsum(), [0], 0)
    tracks = np.concatenate(tracks)

    segments_toc = np.array(list(map(len, segments)))
    segments_toc = np.insert(segments_toc.cumsum(), [0], 0)
    segments = np.concatenate(segments)

    ofname = os.path.splitext(os.path.basename(data_file))[0] + '.npz'
    np.savez_compressed(
        ofname, 
        summary=summary, 
        tracks=tracks, tracks_toc=tracks_toc,
        segments=segments, segments_toc=segments_toc,
    )

    shutil.move(ofname, '/sdf/group/neutrino/kvtsang/slacube/run3')

if __name__ == '__main__':
    fire.Fire(main)
