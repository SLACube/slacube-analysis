import h5py
import numpy as np

from slacube.utils import strptime_from_file, group_by_time

def analyze_packet_rate(fpath, duration, tz):
    with h5py.File(fpath, 'r') as f:
        pkts = f['packets']
        file_ts = strptime_from_file(f.filename, tz=tz)
        
        time_bins, slices = group_by_time(pkts, duration)
        nbins = len(slices)

        pkt_cnts = np.zeros((nbins, 8), dtype=int)
        frac_parity_err = np.zeros(nbins, dtype=float)
        frac_invalid_id = np.zeros_like(frac_parity_err)
        
        for t, selected in enumerate(slices):        
            # count packet type
            pkt_types = pkts['packet_type'][selected]
            utype, cnts = np.unique(pkt_types, return_counts=True)
            pkt_cnts[t, utype] = cnts

            # fraction of parity err (data word only)
            data_pkts = pkts[selected][pkt_types == 0]
            frac_parity_err[t] = np.count_nonzero(
                data_pkts['valid_parity'] != 1
            ) / len(data_pkts)

            # fraction of invalid id
            chip_ids = data_pkts['chip_id']
            channels = data_pkts['channel_id']
            frac_invalid_id[t] = np.count_nonzero(
                (chip_ids < 10) | (chip_ids > 110)
                | (channels < 0) | (channels > 63)
            ) / len(data_pkts)
            
    return {
        'timestamp' : file_ts + time_bins[1:] - time_bins[0],
        'duration' : np.diff(time_bins),
        'pkt_cnts' : pkt_cnts,
        'frac_parity_err' : frac_parity_err,
        'frac_invalid_id' : frac_invalid_id,
    }   
