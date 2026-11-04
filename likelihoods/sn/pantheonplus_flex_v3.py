import os
import numpy as np
from cobaya.likelihoods.base_classes import SN
from astropy.cosmology import w0waCDM
from scipy.interpolate import interp1d

class PantheonPlus_flex_v3(SN):
    """
    Likelihood for Pantheon+ (without SH0ES) type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077
    """

    def init_params(self, ini):
        self.twoscriptmfit = False
        data_file = os.path.normpath(os.path.join(self.path, ini.string("data_file")))
        self._read_data_file(data_file)
        self.covs = {}
        for name in ['mag']:
            self.log.debug('Reading covmat for: %s ' % name)
            self.covs[name] = self._read_covmat(
                os.path.join(self.path, ini.string('%s_covmat_file' % name)))
        self.alphabeta_covmat = False
        self.configure()
        self.inverse_covariance_matrix()
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False

    def _apply_mask(self, zmask):
        for col in self.cols:
            setattr(self, col, getattr(self, col)[zmask])
        for name, cov in self.covs.items():
            self.covs[name] = cov[np.ix_(zmask, zmask)]

    def configure(self):
        self._apply_mask(zmask=self.zcmb > 0.01)
        self.pre_vars = 0.  # diagonal component

    def _read_cols(self, data_file, file_cols, sep=None):
        self.log.debug('Reading %s' % data_file)
        with open(data_file, 'r') as f:
            lines = f.readlines()
            line = lines[0]
            if line.startswith('#'):
                line = line[1:]
            cols = [col.strip().lower() for col in line.split(sep)]
            assert cols[0].isalpha()
            indices = [cols.index(col) for col in file_cols]
            zeros = np.zeros(len(lines) - 1)
            for col in self.cols:
                setattr(self, col, zeros.astype(dtype='f8', copy=True))
            for ix, line in enumerate(lines[1:]):
                vals = [val.strip() for val in line.split(sep)]
                vals = [vals[i] for i in indices]
                for i, (col, val) in enumerate(zip(self.cols, vals)):
                    tmp = getattr(self, col)
                    tmp[ix] = np.asarray(val, dtype=tmp.dtype)
        self.nsn = ix + 1
        self.log.debug('Number of SN read: %s ' % self.nsn)

    def _read_data_file(self, data_file):
        file_cols = ['m_b_corr', 'zhd', 'zhel']
        self.cols = ['mag', 'zcmb', 'zhel']
        self._read_cols(data_file, file_cols)

    def _marginalize_abs_mag(self):
        deriv = np.ones_like(self.mag)[:, None]
        derivp = self.invcov.dot(deriv)
        fisher = deriv.T.dot(derivp)
        self.invcov = self.invcov - derivp.dot(np.linalg.solve(fisher, derivp.T))

    def logp(self, **params_values):
        slope = params_values.get('slope', None)
        if slope == 0: 
            correction = 0
        else:
            correction = slope * self.AgeBias(self.zcmb)
        angular_diameter_distances = self.provider.get_angular_diameter_distance(self.zcmb)
        lumdists = (5 * np.log10((1 + self.zhel) * (1 + self.zcmb) * angular_diameter_distances)) + correction

        if self.use_abs_mag:
            Mb = params_values.get('Mb', None)
        else:
            Mb = 0
        if self.marginalize:
            # Should parallelize this loop
            for i in range(self.int_points):
                self.marge_grid[i] = - self.alpha_beta_logp(
                    lumdists, self.alpha_grid[i],
                    self.beta_grid[i], Mb,
                    invcovmat=self.invcovs[i])
            grid_best = np.min(self.marge_grid)
            return - grid_best + np.log(
                np.sum(np.exp(- self.marge_grid[self.marge_grid != np.inf] + grid_best)) *
                self.step_width_alpha * self.step_width_beta)
        else:
            if self.alphabeta_covmat:
                return self.alpha_beta_logp(lumdists, params_values[self.alpha_name],
                                            params_values[self.beta_name], Mb)
            else:
                return self.alpha_beta_logp(lumdists, Mb=Mb)
    
    def alpha_beta_logp(self, lumdists, Mb=0., **kwargs):
        if self.use_abs_mag:
            estimated_scriptm = Mb + 25
        else:
            estimated_scriptm = 0.
        diffmag = self.mag - lumdists - estimated_scriptm
        return - diffmag.dot(self.invcov).dot(diffmag) / 2.
    
    #####################################################
    ################ PARAMETER SETUP &   ################       
    ################ AGE-BIAS CORRECTION ################  
    #####################################################

    def get_requirements(self):
        # State requisites to the theory code
        z_end        = 10 + 0.05
        z_step       = 0.05 
        z_list_inter = np.arange(0, z_end, z_step)
        z_list_inter = np.concatenate((z_list_inter , np.array([20, 100, 1000])))
        z_list_inter = np.concatenate((z_list_inter, self.zcmb))
        reqs = {"angular_diameter_distance": {"z": self.zcmb}, "physical_time": {"z":z_list_inter}}
        if self.use_abs_mag:
            reqs["Mb"] = None
        return reqs

    def get_can_support_params(self):
        params = ['slope']
        return params

    def CSFR(self, z):
        A      = -0.997
        B      = 0.241
        C      = 0.180
        z_0    = 1.243
        result = C / (10 ** (A * (z - z_0)) + 10 ** (B * (z - z_0)))
        return result
    
    def DTD(self, t):
        t_p   = 0.3
        s     = -1.0
        alpha = 20.0
        return ((t / t_p) ** alpha) / ((t / t_p) ** (alpha - s) + 1)

    def AgeBias(self, z):
        z_end        = 10 + 0.05
        z_step       = 0.05 
        z_list_inter = np.arange(0, z_end, z_step)
        z_list_inter = np.concatenate((z_list_inter , np.array([20, 100, 1000])))
        epoch_list   = self.provider.get_physical_time(z_list_inter)
        f1           = interp1d(epoch_list, z_list_inter)

        median_progenitor_respect2z0 = []
        dt = 10 ** -2
        T = self.provider.get_physical_time(1000)

        for z0 in [0]:
            t0        = self.provider.get_physical_time(z0)
            tau       = np.arange(0, t0-T, dt)
            CSFR_list = self.CSFR(f1(t0-tau))
            DTD_list  = self.DTD(tau)
            SPAD_list = CSFR_list * DTD_list     
            AgeNow    = tau[np.searchsorted(np.cumsum(SPAD_list) / np.sum(SPAD_list), 0.5)]
        
        for z0 in z:
            t0        = self.provider.get_physical_time(z0)
            tau       = np.arange(0, t0-T, dt)
            CSFR_list = self.CSFR(f1(t0-tau))
            DTD_list  = self.DTD(tau)
            SPAD_list = CSFR_list * DTD_list  
            AgeThen   = tau[np.searchsorted(np.cumsum(SPAD_list) / np.sum(SPAD_list), 0.5)]
            median_progenitor_respect2z0.append(AgeNow - AgeThen)

        median_progenitor_respect2z0 = np.array(median_progenitor_respect2z0)
		
        return median_progenitor_respect2z0
 
    

