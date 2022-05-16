import h5py

class AdcCalib:
    def __init__(self, peds, vdda=1800, vref_dac=185, vcm_dac=41, gain=0.25):
        self._vref = vdda * vref_dac / 256.
        self._vcm = vdda * vcm_dac / 256.
        self._gain = gain
        self._peds = peds
    
    def convert(self, uids, adcs, ped_sub=True, as_adc=False):
        charge = adcs.astype(float)

        if ped_sub:
            charge -= self._peds[uids]

        if as_adc:
            return charge

        charge *= self._gain * (self._vref - self._vcm) / 256.
        return charge
    
    def __call__(self, pkts, uids=None, **kwargs):
        if uids is None:
            uids = get_ch_uids(pkts)
        adcs = pkts['dataword']
        return self.convert(uids, adcs, **kwargs)
    
    @classmethod
    def load(cls, fpath, **kwargs):
        with h5py.File(fpath, 'r') as f:
            peds = f['pedestal']

            active = peds['active']
            _peds = peds['mean'].copy()
            _peds[~active] = peds[active]['mean'].mean()

        return cls(_peds, **kwargs)
