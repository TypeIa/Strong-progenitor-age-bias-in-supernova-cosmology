import os
import numpy as np
from .pantheonplus_flex_v3 import PantheonPlus_flex_v3
from astropy.cosmology import w0waCDM
from scipy.interpolate import interp1d

class DESDovekie_flex_v3(PantheonPlus_flex_v3):
    """
    Likelihood for DES-Dovekie type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2511.07517
    """

    def _read_data_file(self, data_file):
        file_cols = ['idsurvey', 'zhd', 'zhel', 'mu', 'muerr', 'muerr_vpec', 'muerr_sys', 'probia_beams']
        self.cols = ['idsurvey', 'zcmb', 'zhel', 'mag', 'magerr', 'magerr_vpec', 'magerr_sys', 'probia_beams']
        self._read_cols(data_file, file_cols, sep=",")

        
    def init_params(self, ini):

        self.twoscriptmfit = False
        data_file = os.path.normpath(os.path.join(self.path, ini.string("data_file")))
        self._read_data_file(data_file)
        self.covs = {}
        for name in ["mag"]:
            self.log.debug("Reading inverse covmat for: %s " % name)
            self.invcov = self._read_inv_covmat(
                os.path.join(self.path, ini.string("%s_covmat_file" % name))
            )
        self.alphabeta_covmat = False
        self.configure()
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False
