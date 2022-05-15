import yaml
import numpy as np

MAX_UID = 100 * 64
def load_geom():
    with open('/sdf/group/neutrino/larpix/etc/layouts/layout-2.4.0.yaml', 'r') as f:
        geo = yaml.safe_load(f)
    
    chip_pix = dict([(chip_id, pix) for chip_id,pix in geo['chips']])
    
    pix_loc = np.full((MAX_UID, 2), np.nan, dtype=float)
    for chip_id, pix_ids in chip_pix.items():
        for ch, pix_id in enumerate(pix_ids):
            if pix_id is None:
                continue
            uid = (chip_id - 11 << 6) + ch
            pix_loc[uid] = geo['pixels'][pix_id][1:3]
    
    return pix_loc
