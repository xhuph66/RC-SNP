import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score

from .reservoir_NSNP import ReservoirNSNP


class RC_SNP_model(object):

    def __init__(self,
                 # reservoir
                 reservoir=None,
                 reservoir_size=None,
                 spectral_radius=None,
                 connectivity=None,
                 input_scaling=None,
                 noise_level=None,
                 n_drop=None,
                 # NSNP
                 alpha=1.0,
                 beta=1.0,
                 # representation
                 mts_type=None,
                 reg_ridge_embedding=None,
                 # readout
                 readout_type=None,
                 reg_ridge=None,
                 ):

        self.n_drop = n_drop
        self.alpha = alpha
        self.beta = beta
        self.mts_type = mts_type
        self.readout_type = readout_type

        # Initialize reservoir
        if reservoir is None:
            self._reservoir = ReservoirNSNP(reservoir_size=reservoir_size,
                                            spectral_radius=spectral_radius,
                                            connectivity=connectivity,
                                            input_scaling=input_scaling,
                                            noise_level=noise_level)
        else:
            self._reservoir = reservoir

        # Initialize ridge regression model
        if mts_type == 'reservoir':
            self._ridge_embedding = Ridge(alpha=reg_ridge_embedding, fit_intercept=True)

        # Initialize readout type            
        if self.readout_type is not None:
            if self.readout_type == 'lin':
                self.readout = Ridge(alpha=reg_ridge)
            else:
                raise RuntimeError('Invalid readout type')

    def get_RC_NSNP_states(self, X):
        """ Compute reservoir_NSNP states """
        res_states = self._reservoir.get_states(X, n_drop=self.n_drop, alpha=self.alpha, beta=self.beta)
        return res_states

    def representation_MTS(self, X, res_states):
        """ Generate representation of the MTS """
        coeff_tr = []
        biases_tr = []

        # Reservoir model space representation
        if self.mts_type == 'reservoir':
            for i in range(X.shape[0]):
                u_t, u_t1 = res_states[i, 0:-1, :], res_states[i, 1:, :]
                self._ridge_embedding.fit(u_t, u_t1)
                coeff_tr.append(self._ridge_embedding.coef_.ravel())
                biases_tr.append(self._ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)

        # Last state representation
        elif self.mts_type == 'last':
            input_repr = res_states[:, -1, :]

        else:
            raise RuntimeError('Invalid representation ID')
        return input_repr

    def compute_scores(self, predict_class, Yte):
        """ Wrapper to compute classification accuracy """
        true_class = np.argmax(Yte, axis=1)
        accuracy = accuracy_score(true_class, predict_class)
        return accuracy

    def train(self, X, Y=None):
        time_start = time.time()

        res_states = self.get_RC_NSNP_states(X)
        input_repr = self.representation_MTS(X, res_states)
        if self.readout_type is None:
            self.input_repr = input_repr
        elif self.readout_type == 'lin':
            self.readout.fit(input_repr, Y)

        tot_time = (time.time() - time_start) / 60
        return tot_time

    def test(self, Xte, Yte):
        res_states_te = self.get_RC_NSNP_states(Xte)
        input_repr_te = self.representation_MTS(Xte, res_states_te)
        if self.readout_type == 'lin':
            logits = self.readout.predict(input_repr_te)
            predict_class = np.argmax(logits, axis=1)

        accuracy = self.compute_scores(predict_class, Yte)

        return accuracy
