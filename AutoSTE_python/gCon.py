import numpy as np



def gCon(theta):



    if isinstance(theta, dict):
        gVal = np.ones((len(theta['Q']), 4), dtype=bool)
        gVal[:, 0] = theta['Q'] >= 0
        gVal[:, 1] = theta['u'] >= 0
        gVal[:, 2] = theta['ci'] > 0
        gVal[:, 3] = theta['cii'] > 0

        consTrue = np.prod(gVal, axis=1).astype(int)
    else:
        gVal = np.ones((theta.shape[1], 4), dtype=bool)
        gVal[:, 0] = theta[3, :] >= 0
        gVal[:, 1] = theta[4, :] >= 0
        gVal[:, 2] = theta[6, :] > 0
        gVal[:, 3] = theta[7, :] > 0

        consTrue = np.prod(gVal, axis=1).astype(int)

        is_first_call = False

    return consTrue
