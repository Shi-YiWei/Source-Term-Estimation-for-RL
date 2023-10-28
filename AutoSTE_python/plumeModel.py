import numpy as np


def plumeModel(s, p):
    # Plume dispersion model
    # s is the source term (either a structure or array)
    # p contains the locations of the concentration readings
    # C is the concentration level from the plume model

    # Dispersion model assumes isotropic diffusion from the source
    # Taken from Vergassola et al. (2007) and also in Hutchinson et al. (2018)
    # See IP model in https://onlinelibrary.wiley.com/doi/epdf/10.1002/rob.21844

    thresh = 5e-5
    value = 0

    if isinstance(s, dict):
        D = s['ci']
        t = s['cii']
        lamda = np.sqrt((D * t) / (1 + (s['u'] ** 2 * t) / (4 * D)))

        module_dist = np.sqrt(
            (s['x'] - p['x_matrix']) ** 2 + (s['y'] - p['y_matrix']) ** 2 + (s['z'] - p['z_matrix']) ** 2)
        module_dist = np.where(module_dist < thresh, value, module_dist)

        C = s['Q'] / (4 * np.pi * D * module_dist) * np.exp(
            (-(p['x_matrix'] - s['x']) * s['u'] * np.cos(s['phi']) / (2 * D)) +
            (-(p['y_matrix'] - s['y']) * s['u'] * np.sin(s['phi']) / (2 * D)) +
            (-module_dist / lamda)
        )


    else:
        x = s[0, :]
        y = s[1, :]
        z = s[2, :]
        Q = s[3, :]
        u = s[4, :]
        phi = s[5, :]
        D = s[6, :]
        t = s[7, :]

        lamda = np.sqrt((D * t) / (1 + (u ** 2 * t) / (4 * D)))

        module_dist = np.sqrt((x - p['x_matrix']) ** 2 + (y - p['y_matrix']) ** 2 + (z - p['z_matrix']) ** 2)
        module_dist = np.where(module_dist < thresh, value, module_dist)

        C = Q / (4 * np.pi * D * module_dist) * np.exp(
            (-(p['x_matrix'] - x) * u * np.cos(phi) / (2 * D)) +
            (-(p['y_matrix'] - y) * u * np.sin(phi) / (2 * D)) +
            (-module_dist / lamda)
        )

    return C
