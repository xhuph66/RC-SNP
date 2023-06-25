"""
RC_SNP model configuration and hyperparameter values file
"""


def config_datasets():
    config = {}

    config['seed'] = 1

    # Hyperarameters of the reservoir_SNP
    config['reservoir_size'] = 800  # size of the reservoir_NSNP
    config['spectral_radius'] = 0.99  # spectral radius of the reservoir_NSNP
    config['alpha'] = 0.2  # RC_SNP state update equation parameters ¦Á(0 --> 1.0)
    config['beta'] = 0.9  # RC_SNP state update equation parameters ¦Â(0 --> 1.0)
    config['connectivity'] = 0.25  # percentage of nonzero connections in the reservoir_NSNP
    config['input_scaling'] = 0.15  # scaling of the input weights
    config['noise_level'] = 0.001  # noise in the reservoir_NSNP state update
    config['n_drop'] = 5  # transient states to be dropped

    # Type of MTS representation
    config['mts_type'] = 'reservoir'  # MTS representation:  {'last','reservoir'}
    config['reg_ridge_embedding'] = 5.0  # regularization parameter of the ridge regression

    # Type of readout
    config['readout_type'] = 'lin'
    config['reg_ridge'] = 1.0  # regularization of the ridge regression readout

    return config
