import numpy as np
from scipy import sparse


class ReservoirNSNP(object):
    """
    Build the reservoir constructed by the NSNP system and calculate the output of the internal state
    
    Parameters:
        reservoir_size = Number of neurons in the reservoir_NSNP
        spectral_radius = The maximum value of the eigenvalue mode of the connection weight
            matrix of neurons in the reservoir_NSNP layer
        connectivity = Percentage of non-zero connection weights between neurons in the reservoir_NSNP
        input_scaling = Scaling of the connection weights between the input neurons and the reservoir_NSNP neurons
        noise_level = Gaussian noise introduced in the state update of the reservoir_NSNP
    """

    def __init__(self, reservoir_size=100, spectral_radius=0.99, connectivity=0.3, input_scaling=0.2, noise_level=0.01):
        """ Initialize reservoir properties """
        self._reservoir_size = reservoir_size
        self._input_scaling = input_scaling
        self._noise_level = noise_level

        # Input weights are determined by the size of the input data
        self._input_weights = None

        # Generate reservoir internal connection weights
        self._reservoir_weights = self._initialize_reservoir_weights(reservoir_size, connectivity, spectral_radius)

    def _initialize_reservoir_weights(self, reservoir_size, connectivity, spectral_radius):
        """ Initialize the reservoir internal connection weight matrix """

        # Generate uniformly distributed and sparse reservoir internal weights
        reservoir_weights = sparse.rand(reservoir_size, reservoir_size, density=connectivity).todense()

        # Ensure that the values of the reservoir internal weights are uniformly distributed in the range of [-0.5, 0.5]
        reservoir_weights[np.where(reservoir_weights > 0)] -= 0.5

        # Adjust the spectral radius.
        E, _ = np.linalg.eig(reservoir_weights)
        e_max = np.max(np.abs(E))
        reservoir_weights /= np.abs(e_max) / spectral_radius

        return reservoir_weights

    def _compute_state_matrix_NSNP(self, X, n_drop=0, alpha=0, beta=0):
        """ Status update of reservoirs constructed by NSNP systems """
        N, T, _ = X.shape
        previous_state = np.zeros((N, self._reservoir_size), dtype=float)
        state_matrix = np.empty((N, T - n_drop, self._reservoir_size), dtype=float)

        for t in range(T):
            current_input = X[:, t, :]

            state_before_tanh = beta * previous_state.T + self._input_weights.dot(current_input.T)
            # Add noise
            state_before_tanh += np.random.rand(self._reservoir_size, N) * self._noise_level
            previous_state = alpha * previous_state + self._reservoir_weights.dot(np.tanh(state_before_tanh)).T

            # Reserved everything after the dropout period
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state

        return state_matrix

    def get_states(self, X, n_drop=0, alpha=0, beta=0):
        N, T, V = X.shape
        if self._input_weights is None:
            # Generate weights from -0.1 and 0.1 binomial distributions.
            self._input_weights = (2.0 * np.random.binomial(1, 0.5,
                                                            [self._reservoir_size, V]) - 1.0) * self._input_scaling

        # Sequence for calculating NSNP reservoir state
        states = self._compute_state_matrix_NSNP(X, n_drop, alpha, beta)

        return states
