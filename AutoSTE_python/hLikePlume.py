import numpy as np
from plumeModel import plumeModel
from scipy.special import erf


def hLikePlume(theta, sensor_data, pos, m):
    # Likelihood function based on the plume model

    conc = plumeModel(theta, pos)

    sigma0 = m['thresh']
    sigmaN = m['sig_pct'] * conc + m['sig']

    if sensor_data <= m['thresh']:
        likelihood = m['Pd'] * 1/2 * (1 + erf((m['thresh'] - conc) / (sigma0 * np.sqrt(2)))) + (1 - m['Pd'])
    else:
        likelihood = 1 / (sigmaN * np.sqrt(2 * np.pi)) * np.exp(-(sensor_data - conc)**2 / (2 * sigmaN**2))

    return likelihood
