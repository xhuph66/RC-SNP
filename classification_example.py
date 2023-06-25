import random
import time

import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder
from RC_SNP import RC_SNP_model
from RC_SNP import configurations

# Parameter introduction of RC_SNP model
config = configurations.config_datasets()
config['dataset_name'] = 'CMU'
print(config)
# Set and fix random seeds
np.random.seed(config['seed'])
random.seed(config['seed'])

# ============ Load dataset ============
data = scipy.io.loadmat('dataset/' + config['dataset_name'] + '.mat')
Xtr, Xte = data['X'], data['Xte']  # shape is [N,T,V]
if len(Xtr.shape) < 3:
    Xtr = np.atleast_3d(Xtr)
if len(Xte.shape) < 3:
    Xte = np.atleast_3d(Xte)
Ytr, Yte = data['Y'], data['Yte']  # shape is [N,1]

print('\nLoaded ' + config['dataset_name'] + ' - Tr: ' + str(Xtr.shape) + ', Te: ' + str(Xte.shape))

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse=False)
Ytr = onehot_encoder.fit_transform(Ytr)
Yte = onehot_encoder.transform(Yte)

accuracys = []

# ============ Initialize, train and evaluate the RC_SNP model ============
time_start = time.time()
for i in range(0, 10):
    each_random_time_start = time.time()
    print('=========Rand = %d=========' % (i + 1))
    classifier = RC_SNP_model(
        reservoir=None,
        reservoir_size=config['reservoir_size'],
        spectral_radius=config['spectral_radius'],
        connectivity=config['connectivity'],
        input_scaling=config['input_scaling'],
        noise_level=config['noise_level'],
        n_drop=config['n_drop'],
        alpha=config['alpha'],
        beta=config['beta'],
        mts_type=config['mts_type'],
        reg_ridge_embedding=config['reg_ridge_embedding'],
        readout_type=config['readout_type'],
        reg_ridge=config['reg_ridge'],
    )
    tr_time = classifier.train(Xtr, Ytr)
    print('Training time = %.2f mins' % tr_time)

    accuracy = classifier.test(Xte, Yte)

    each_random_tot_time = (time.time() - each_random_time_start) / 60
    print('Accuracy = %.4f-----Each Random Time =%.2f mins' % (accuracy, each_random_tot_time))
    accuracys.append(accuracy)

mean_acc = np.mean(accuracys) * 100
std_acc = np.std(accuracys) * 100

tot_time = (time.time() - time_start) / 60
mean_time = tot_time / 10
print('\ndataset : %s  accuracy = %.2f±%.2f  Mean Time =%.2f mins\n' % (
    config['dataset_name'], mean_acc, std_acc, mean_time))
