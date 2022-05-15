import h5py
import numpy as np

from tqdm import tqdm
from slacube.utils import filter_data_packets, group_by_time
from slacube.geom import MAX_UID

def analyze_pedestal(pkts, trunc=True, return_hist=False, show_progress=False):
    mask, uids = filter_data_packets(pkts, return_uids=True)
    data_pkts = pkts[mask]
    
    summary = np.zeros(
        MAX_UID,
        dtype=[('active',bool), ('mean',float), ('std', float)]
    )
    hist = None

    for uid in tqdm(np.unique(uids), disable=not show_progress):
        mask = uids == uid
        if np.count_nonzero(mask) < 5:
            continue
            
        data = data_pkts[mask]
        ped = data['dataword']
        
        if trunc or return_hist:
            cnts = np.histogram(ped, bins=256, range=(0,256))[0]
            

        if trunc:
            peak = np.argmax(cnts)
            start = max(0, peak-3)
            stop = min(256, peak+4)
            
            values = np.arange(start, stop).astype(float)
            weights = cnts[start:stop]
            
            mean = np.average(values, weights=weights)
            var = np.average((values - mean)**2, weights=weights)
            std = np.sqrt(var)
        else:
            mean = ped.mean()
            std = ped.std()

        summary[uid] = True, mean, std
        
        if return_hist:
            if hist is None:
                hist = np.zeros((MAX_UID, 256), dtype=int)
            hist[uid] = cnts
        
    output = [summary]
    if return_hist:
        output.append(hist)
    
    if len(output) == 1:
        return summary

    return tuple(output)

def monitor_pedestal(pkts, duration, hist_mode=False, show_progress=False):
    time_bins, slices = group_by_time(pkts, duration)

    output = []
    for s in tqdm(slices, disable=not show_progress):
        ped = analyze_pedestal(pkts[s])

        if hist_mode:
            mask = ped['active']
            hist = np.histogram(ped[mask]['mean'], bins=256, range=(0,256))[0]
            output.append(hist)
        else:
            output.append(ped)

    ts = time_bins[:-1]
    output = np.asarray(output)

    return output, ts
