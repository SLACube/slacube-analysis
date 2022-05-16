import os
import h5py
import numpy as np
from datetime import datetime

def filter_data_packets(pkts, return_uids=False):
    chip_ids = pkts['chip_id']
    channels = pkts['channel_id']
    
    mask = (pkts['valid_parity'] == 1) & (pkts['packet_type'] == 0)
    
    # remove invalid IDs (corrupted data?)
    mask &= (chip_ids >= 11) & (chip_ids <= 110)
    mask &= (channels >= 0) & (channels <= 63)
    
    output = [mask]
    
    if return_uids:
        uids = gen_ch_uids(chip_ids[mask], channels[mask])
        output.append(uids)
    
    if len(output) == 1:
        return output[0]
    return tuple(output)

def match_unix_timestamp(pkts, last_unix_ts):
    mask = filter_data_packets(pkts)
    mask |= pkts['packet_type'] == 4
    
    buf = np.copy(pkts[mask])
    buf = np.insert(buf, [0], last_unix_ts)
    
    ts_mask = buf['packet_type'] == 4
    ts_grps = np.split(buf, np.argwhere(ts_mask).flat)
    unix_ts = np.concatenate(
        [[ts_grp[0]]*len(ts_grp[1:]) for ts_grp in ts_grps if len(ts_grp) > 1],
        axis=0
    )
    
    return buf[~ts_mask], unix_ts

def get_ch_uids(pkts):
    chip_ids = pkts['chip_id']
    channels = pkts['channel_id']
    return gen_ch_uids(chip_ids, channels)

def gen_ch_uids(chip_ids, channels):
    uids = ((chip_ids.astype(int) - 11) << 6) + channels.astype(int)
    return uids

def strptime_from_file(fpath, tz):
    fname = os.path.basename(fpath)
    fname = fname[fname.find('_')+1:fname.find(f'_{tz}')]
    file_ts = datetime.strptime(fname, '%Y_%m_%d_%H_%M_%S').timestamp()

    return file_ts

def group_by_time(pkts, duration):
    unix_ts = pkts[pkts['packet_type'] == 4]['timestamp']
    idx_pkt4 = np.where(pkts['packet_type'] == 4)[0]
    
    t_min = unix_ts.min()
    t_max = unix_ts.max()
    nbins = max(1, int((t_max - t_min) / duration))
    time_bins = np.linspace(t_min, t_max, nbins+1)
    
    slices = []
    for t in range(nbins):
        start = np.where(unix_ts >= time_bins[t])[0].min()
        stop = np.where(unix_ts < time_bins[t+1])[0].max()
        slices.append(slice(idx_pkt4[start], idx_pkt4[stop+1]))
    
    return time_bins, slices
