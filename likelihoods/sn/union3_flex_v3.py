import os
import numpy as np
from .pantheonplus_flex_v3 import PantheonPlus_flex_v3
from astropy.cosmology import w0waCDM
from scipy.interpolate import interp1d

class Union3_flex_v3(PantheonPlus_flex_v3):
    """
    Likelihood for the Union3 & UNITY1.5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/pdf/2311.12098.pdf
    """

    def configure(self):
        self.pre_vars = 0.  # diagonal component

    def _read_data_file(self, data_file):
        file_cols = ['zcmb', 'mb']
        self.cols = ['zcmb', 'mag']
        self._read_cols(data_file, file_cols)
        self.zhel = self.zcmb
