import numpy as np
import time
from resamplingIndex import resampling_index
import numpy as np
from gCon import gCon
from scipy.linalg import block_diag


def ESS(w):
    # Calculate the Effective Sample Size (ESS) from weights
    sumw2 = np.sum(w ** 2)
    ess = 1 / sumw2
    return ess


def mcmcPF(theta, w_k_i, sensor_data, fDyn, fParm, hLike, sensor_model_parameters, pos, gParm, gCon=None):
    # Perform the MCMC Particle Filter (MCMC-PF)

    ct = time.time()  # overall calculation time
    N = len(w_k_i)  # number of particles
    n = 8  # length(fieldnames(xpartminus))

    xpart = theta.copy()  # propagation of state

    wupdate = hLike(xpart, sensor_data, pos, sensor_model_parameters)  # likelihood update

    if gCon is None:
        wnew = w_k_i * wupdate

    else:
        wcon = gCon(xpart)
        wnew = w_k_i * wupdate * wcon

    wnew = wnew / np.sum(wnew)  # normalization

    len_num = len(wnew)

    # print(wnew.T.reshape(20000, 1))

    ess = ESS(wnew)  # effective sample size

    if ess < 0.5 * N:
        State = np.vstack(
            (xpart['x'], xpart['y'], xpart['z'], xpart['Q'], xpart['u'], xpart['phi'], xpart['ci'], xpart['cii']))

        # State = [xpart['x'], xpart['y'], xpart['z'], xpart['Q'], xpart['u'], xpart['phi'], xpart['ci'], xpart['cii']]

        # avgState = np.sum(State * wnew.reshape(-1, 1), axis=1)

        avgState = np.sum(np.dot(np.ones((n, 1)), wnew.T.reshape(1, len_num)) * State, axis=1)

        # (State(1:3,:) - avgState(1:3)*ones(1,N))*diag(wnew)*(State(1:3,:) - avgState(1:3)*ones(1,N))';

        covPos = (State[0:3, :] - avgState[0:3].reshape(3, 1) @ np.ones((1, N))) @ np.diag(wnew) @ (
                State[0:3, :] - avgState[0:3].reshape(3, 1) @ np.ones((1, N))).T
        covQ = (State[3, :] - avgState[3].reshape(1, 1) @ np.ones((1, N))) @ np.diag(wnew) @ (
                State[3, :] - avgState[3].reshape(1, 1) @ np.ones((1, N))).T
        covWind = (State[4:6, :] - avgState[4:6].reshape(2, 1) @ np.ones((1, N))) @ np.diag(wnew) @ (
                State[4:6, :] - avgState[4:6].reshape(2, 1) @ np.ones((1, N))).T
        covDiff = (State[6:8, :] - avgState[6:8].reshape(2, 1) @ np.ones((1, N))) @ np.diag(wnew) @ (
                State[6:8, :] - avgState[6:8].reshape(2, 1) @ np.ones((1, N))).T

        Dpos = np.linalg.cholesky(covPos)
        Dq = np.linalg.cholesky(covQ)
        Dwind = np.linalg.cholesky(covWind)
        Ddiff = np.linalg.cholesky(covDiff)

        wnew, index = resampling_index(wnew)
        State = State[:, index]

        # wnew = wnew.T

        A = (4 / (n + 2)) ** (1 / (n + 4))
        hopt = A * (N ** (-1 / (n + 4)))

        idx_ = np.ones(N, dtype=bool)
        newState = State.copy()

        for _ in range(3):
            idx = idx_
            # print(idx)
            newState[:3, idx] = State[:3, idx] + hopt * Dpos @ np.random.randn(3, np.sum(idx))
            newState[3, idx] = State[3, idx] + hopt * Dq @ np.random.randn(1, np.sum(idx))
            newState[4:6, idx] = State[4:6, idx] + hopt * Dwind @ np.random.randn(2, np.sum(idx))
            newState[6:8, idx] = State[6:8, idx] + hopt * Ddiff @ np.random.randn(2, np.sum(idx))

            # idx = np.logical_not(gCon(newState))
            idx = np.where(gCon(newState) != 1)[0]
            if np.sum(idx) == 0:
                break
            else:
                newState[:, idx] = State[:, idx]

        newerr = newState - State

        SIG = hopt ** 2 * block_diag(covPos, covQ, covWind, covDiff)

        logratio = - 0.5 * np.sum(np.dot(newerr.T, np.linalg.inv(SIG)).T * newerr, axis=0) \
                   + 0.5 * np.sum(np.dot(np.zeros((n, N)).T, np.linalg.inv(SIG)).T * np.zeros((n, N)), axis=0)

        xupdate = hLike(State, sensor_data, pos, sensor_model_parameters)
        xnewupdate = hLike(newState, sensor_data, pos, sensor_model_parameters)

        alpha = xnewupdate / xupdate * np.exp(logratio)

        mcrand = np.random.rand(N)
        accept = alpha >= mcrand
        reject = alpha < mcrand
        newState[:, reject] = State[:, reject]

        newpart = {
            'x': newState[0, :],
            'y': newState[1, :],
            'z': newState[2, :],
            'Q': newState[3, :],
            'u': newState[4, :],
            'phi': newState[5, :],
            'ci': newState[6, :],
            'cii': newState[7, :]
        }

    else:
        newpart = xpart

    time_taken = time.time() - ct

    xest = None  # Not sure where this is used in your code, so leaving it as None
    info = {
        'ess': ess,
        'avgSampling': 0,  # Not sure where this is calculated, so leaving it as 0
        'time': time_taken
    }

    return newpart, wnew, info
