import h5py
import numpy as np
from tqdm import trange
from sklearn.cluster import dbscan
from sklearn.decomposition import PCA

from slacube.evt import SymmetricWindowEventBuilder
from slacube.utils import (
    match_unix_timestamp, filter_data_packets, filter_data_packets
)


def select_tracks(events, timestamps, pix_loc, adc2charge, detprop, cfg):
    
    tracks = []
    segments = []
    summary = []
    cfg_evt = cfg.get('event', {})
    cfg_trk = cfg.get('track', {})
    cfg_seg = cfg.get('segment', {})
    valid_uids = np.all(~np.isnan(pix_loc), axis=1)
    
    for i_ev in range(len(events)):
        mask, ch_uids = filter_data_packets(events[i_ev], return_uids=True)

        mask &= valid_uids[ch_uids]

        evt = events[i_ev][mask]
        evt_ts = timestamps[i_ev][mask]
        ch_uids = ch_uids[mask]

        if len(evt) > cfg_evt.get('max_nhits', np.inf):
            continue

        if len(evt) < cfg_evt.get('min_nhits', 0):
            continue

        # sync_noise_cut
        if evt['timestamp'].min() < 100000:
            continue

        # convert pixel to coord
        dt = evt['timestamp'] - evt['timestamp'].min()
        z = dt * detprop['v_drift'] * detprop['clock_cycle']
        coords = np.column_stack([pix_loc[ch_uids], z])
        
        # dbscan
        clust_ids = dbscan(coords, **cfg_evt['dbscan'])[1]
        clust_uids = np.unique(clust_ids[clust_ids != -1])

        # reject unclustered event
        if len(clust_uids) == 0:
            continue
        
        for c_uid in clust_uids:
            mask = clust_ids == c_uid
            n_hits = np.count_nonzero(mask)
            
            if n_hits < cfg_trk.get('min_nhits', 0):
                continue

            clust_coords = coords[mask]        
            output = analyze_track(clust_coords, **cfg_seg)
            
            hit_frac = output.get('hit_frac', 0)
            if hit_frac < cfg_trk.get('min_hit_frac', 0):
                continue

            seg_cos_th = output.get('segment_cos_theta', None)
            if seg_cos_th is None:
                continue
            if seg_cos_th.min() < cfg_trk.get('min_seg_cos_th', 0):
                continue
                
            adcs = evt['dataword'][mask]
            if adc2charge is None:
                charges = np.zeros(n_hits)
            else:
                charges = adc2charge.convert(ch_uids[mask], adcs)

            x, y, z = clust_coords.T
            t = dt[mask]
            t -= t.min()
            z -= z.min()

            if t.max() < cfg_trk.get('min_dt', 0):
                continue

            if z.max() < cfg_trk.get('min_dz', 0):
                continue

            seg_ids = output['segment_ids']
            trk = np.rec.fromarrays(
                [x, y, z, ch_uids[mask], adcs, charges, evt['timestamp'][mask], seg_ids],
                names='x,y,z,uid,adc,q,dt,segment_id'
            )
            tracks.append(trk)

            dx = output['segment_len']
            n_segs = len(dx)
            charges_fix = charges.copy()
            charges_fix[charges < 0] = 0.
            dq = np.histogram(
                seg_ids, bins=n_segs, range=(0,n_segs), weights=charges_fix)[0]

            seg_nhits = output['segment_nhits']
            t_seg = np.histogram(
                seg_ids, bins=n_segs, range=(0,n_segs), weights=t.astype(float))[0]
            with np.errstate(divide='ignore'):
                t_seg /= seg_nhits
                t_seg = np.nan_to_num(t_seg, posinf=0)

            dx = output['segment_len']

            seg = np.rec.fromarrays(
                [dq,dx,dq/dx,t_seg,seg_nhits,seg_cos_th],
                 names='dq,dx,dqdx,t,n,cos_th',
            )
            segments.append(seg)

            trk_summary = np.empty(
                1,
                dtype=[
                    ('i_ev', int), ('timestamp', int), ('hit_frac',float), 
                    ('seg_cos_th', float), ('len',float), ('theta',float), 
                    ('dz',float), ('dt',float), ('dqdx',float),
                    ('track_size', int), ('segment_size', int),
                ],
            )
            trk_summary['i_ev'] = i_ev
            trk_summary['timestamp'] = evt_ts['timestamp'][mask].min()
            trk_summary['hit_frac'] = hit_frac
            trk_summary['seg_cos_th'] = seg_cos_th.min()
            trk_summary['len'] = output['len']
            trk_summary['theta'] = output['theta']
            trk_summary['dz'] = z.max()
            trk_summary['dt'] = t.max()
            trk_summary['dqdx'] = dq.sum() / output['len']
            trk_summary['track_size'] = len(trk)
            trk_summary['segment_size'] = len(seg)
            summary.append(trk_summary)
    return tracks, segments, summary

def analyze_track(trk, seg_len, min_trk_len, radius, min_seg_nhits=3):
    output = {}
    pca_global = PCA(2)
    xt, yt = pca_global.fit_transform(trk).T
    trk_len = xt.max() - xt.min()
    trk_width = yt.max() - yt.min()
    
    output['len'] = trk_len
    output['theta'] = abs(pca_global.components_[0,2])
    
    # reject short track length
    if trk_len < min_trk_len:
        return output
    
    # divide the track into xx mm segments
    nbins = int(np.floor(trk_len / seg_len))
    bins = np.linspace(xt.min(), xt.max(), nbins+1)
    
    grps = np.digitize(xt, bins)
    grps[grps > nbins] = nbins
    grps -= 1
    
    # loop segments
    step = trk_len / nbins
    seg_lengths = np.full(nbins, step, dtype=float)
    seg_nhits = np.zeros(nbins, dtype=int)
    seg_cos_th = np.zeros(nbins, dtype=float)
    mask_outliner = np.full(len(trk), False)
    
    for i_grp, n_pixels in zip(*np.unique(grps, return_counts=True)):
        seg_nhits[i_grp] = n_pixels
        
        # skip small segment 
        if n_pixels < min_seg_nhits:
            continue

        mask = grps == i_grp
        seg = trk[mask]

        pca = PCA(2)
        xs, ys = pca.fit_transform(seg).T
        
        mask_outliner[mask] = np.abs(ys) < radius

        seg_lengths[i_grp] = xs.max() - xs.min()
        seg_cos_th[i_grp] = np.abs(pca.components_[0].dot(
            pca_global.components_[0]))
        
    grps[~mask_outliner] = -1
    
    output.update({
        'hit_frac' : np.count_nonzero(mask_outliner) / len(trk),
        'segment_ids' : grps,
        'segment_len' : seg_lengths,
        'segment_nhits' : seg_nhits,
        'segment_cos_theta' : seg_cos_th,
    })
    return output

def process_file(fpath, pix_loc, adc2charge=None, n_pkts=None):
    buf_size = 38400
    tracks = []
    segments = []
    summary = []
    
    with h5py.File(fpath, 'r') as f:
        pkts = f['packets']
    
        if n_pkts is not None and n_pkts < len(pkts):
            pkts = pkts[:n_pkts]
        
        builder = SymmetricWindowEventBuilder()
        last_unix_ts = np.array(pkts[pkts['packet_type'] == 4][0], dtype=pkts.dtype)
        n_blocks = int(np.ceil(len(pkts) / buf_size))
        
        for i_blk in trange(n_blocks):
            i0 = i_blk * buf_size
            block = pkts[i0:i0+buf_size]
            
            data_buf, unix_ts = match_unix_timestamp(block, last_unix_ts)
            data_buf['timestamp']  = data_buf['timestamp'].astype(int) % (2**31)
            last_unix_ts = unix_ts[-1]

            events, evt_ts = builder.build_events(data_buf, unix_ts)
            output = select_tracks(events, evt_ts, pix_loc, adc2charge)
            
            tracks.extend(output[0])
            segments.extend(output[1])
            summary.extend(output[2])
    
    summary = np.concatenate(summary)
    return tracks, segments, summary
