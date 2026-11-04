import os
import numpy as np
from .pantheonplus_flex_v3 import PantheonPlus_flex_v3
from astropy.cosmology import w0waCDM
from scipy.interpolate import interp1d

class DESy5_flex_v3(PantheonPlus_flex_v3):
    """
    Likelihood for DES-Y5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2401.02929
    """

    def configure(self):
        self.pre_vars = self.mag_err ** 2

    def _read_data_file(self, data_file):
        file_cols = ['zhd', 'zhel', 'mu', 'muerr_final']
        self.cols = ['zcmb', 'zhel', 'mag', 'mag_err']
        self._read_cols(data_file, file_cols, sep=',')
