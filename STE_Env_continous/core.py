import numpy as np
from scipy.special import erf
from scipy.linalg import block_diag
import time




def plumeModel(s, p, nsg_n):

    np.random.seed(nsg_n)

    # Plume dispersion model
    # s is the source term (either a structure or array)
    # p contains the locations of the concentration readings
    # C is the concentration level from the plume model

    # Dispersion model assumes isotropic diffusion from the source
    # Taken from Vergassola et al. (2007) and also in Hutchinson et al. (2018)
    # See IP model in https://onlinelibrary.wiley.com/doi/epdf/10.1002/rob.21844

    if isinstance(s, dict):
        D = s["ci"]
        t = s["cii"]
        lamda = np.sqrt((D * t) / (1 + (s["u"] ** 2 * t) / (4 * D)))

        module_dist = np.sqrt(
            (s["x"] - p["x_matrix"]) ** 2
            + (s["y"] - p["y_matrix"]) ** 2
            + (s["z"] - p["z_matrix"]) ** 2
        )
        module_dist = np.where(module_dist < 1e-5, 1e-5, module_dist)

        C = (
                s["Q"]
                / (4 * np.pi * D * module_dist)
                * np.exp(
            (-(p["x_matrix"] - s["x"]) * s["u"] * np.cos(s["phi"]) / (2 * D))
            + (-(p["y_matrix"] - s["y"]) * s["u"] * np.sin(s["phi"]) / (2 * D))
            + (-module_dist / lamda)
        )
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

        module_dist = np.sqrt(
            (x - p["x_matrix"]) ** 2
            + (y - p["y_matrix"]) ** 2
            + (z - p["z_matrix"]) ** 2
        )
        module_dist = np.where(module_dist < 1e-5, 1e-5, module_dist)

        C = (
                Q
                / (4 * np.pi * D * module_dist)
                * np.exp(
            (-(p["x_matrix"] - x) * u * np.cos(phi) / (2 * D))
            + (-(p["y_matrix"] - y) * u * np.sin(phi) / (2 * D))
            + (-module_dist / lamda)
        )
        )

    return C


def sensorModel(s, pos, m, nsg):
    nsg_n = nsg.next_seed()
    np.random.seed(nsg_n)
    # Generate simulated sensor data based on the source term, sensor position, and sensor characteristics

    conc = plumeModel(s, pos, nsg_n)

    # Add noise
    # datasize = conc.shape
    #
    # error = m['sig_pct'] * conc * np.random.randn(*datasize)  # Add noise or fluctuations
    #
    # sensorData = conc + error
    #
    # # Not detect if below the threshold
    # sensorData = np.where(sensorData < m['thresh'], 0, sensorData)
    #
    # # Not detect due to the mis-detection rate
    # mask = np.random.rand(*datasize) < (1 - m['Pd'])
    # sensorData = np.where(mask, 0, sensorData)

    return conc #sensorData


def hLikePlume(xpart, yObv, pos, m, nsg):
    # Likelihood function based on the plume model

    conc = plumeModel(xpart, pos, nsg)

    sigma0 = m['thresh']
    sigmaN = m['sig_pct'] * conc + m['sig']

    if yObv <= m['thresh']:
        likelihood = m['Pd'] * 1 / 2 * (1 + erf((m['thresh'] - conc) / (sigma0 * np.sqrt(2)))) + (1 - m['Pd'])
    else:
        likelihood = 1 / (sigmaN * np.sqrt(2 * np.pi)) * np.exp(-(yObv - conc) ** 2 / (2 * sigmaN ** 2))

    return likelihood


def ESS(w):
    # Calculate the Effective Sample Size (ESS) from weights
    sumw2 = np.sum(w ** 2)
    ess = 1 / sumw2
    return ess


def resampling_index(weights, nsg, N=None):
    np.random.seed(nsg.next_seed())
    Ns = len(weights)

    if N is None:
        N = len(weights)
    index = np.zeros(N, dtype=int)
    w = np.zeros(N)
    c = np.cumsum(weights)

    i = 0
    u = np.zeros(N)
    u[0] = np.random.rand() / N
    for j in range(N):
        u[j] = u[0] + (j - 1) / N
        while u[j] > c[i]:
            i += 1
        index[j] = i

    wnew = np.ones(N) / N

    return wnew, index


def resampling_index_np(weights, N=None):
    if N is None:
        N = len(weights)
        index = np.random.choice(N, N, p=weights)
    else:
        index = np.random.choice(len(weights), N, p=weights)
    index = np.sort(index)
    wnew = np.ones(N) / N

    return wnew, index


def mcmcPF(xpartminus, wminus, yobv, fDyn, fParm, hLike, hParm, pos, gParm, nsg, gCon=None):
    # Perform the MCMC Particle Filter (MCMC-PF)

    nsg_n = nsg.next_seed()
    np.random.seed(nsg_n)

    ct = time.time()  # overall calculation time
    N = len(wminus)  # number of particles
    n = 8  # length(fieldnames(xpartminus))

    xpart = xpartminus.copy()  # propagation of state

    wupdate = hLike(xpart, yobv, pos, hParm, nsg_n)  # likelihood update

    if gCon is None:
        wnew = wminus * wupdate

    else:
        wcon = gCon(xpart)
        wnew = wminus * wupdate * wcon

    wnew = wnew / np.sum(wnew)  # normalization

    len_num = len(wnew)

    # print(wnew.T.reshape(20000, 1))

    ess = ESS(wnew)  # effective sample size

    if ess < 0.5 * N:
        State = np.vstack(
            (
                xpart["x"],
                xpart["y"],
                xpart["z"],
                xpart["Q"],
                xpart["u"],
                xpart["phi"],
                xpart["ci"],
                xpart["cii"],
            )
        )

        # State = [xpart['x'], xpart['y'], xpart['z'], xpart['Q'], xpart['u'], xpart['phi'], xpart['ci'], xpart['cii']]

        # avgState = np.sum(State * wnew.reshape(-1, 1), axis=1)

        avgState = np.sum(
            np.dot(np.ones((n, 1)), wnew.T.reshape(1, len_num)) * State, axis=1
        )

        # (State(1:3,:) - avgState(1:3)*ones(1,N))*diag(wnew)*(State(1:3,:) - avgState(1:3)*ones(1,N))';

        covPos = (
                (State[0:3, :] - avgState[0:3].reshape(3, 1) @ np.ones((1, N)))
                @ np.diag(wnew)
                @ (State[0:3, :] - avgState[0:3].reshape(3, 1) @ np.ones((1, N))).T
        )
        covQ = (
                (State[3, :] - avgState[3].reshape(1, 1) @ np.ones((1, N)))
                @ np.diag(wnew)
                @ (State[3, :] - avgState[3].reshape(1, 1) @ np.ones((1, N))).T
        )
        covWind = (
                (State[4:6, :] - avgState[4:6].reshape(2, 1) @ np.ones((1, N)))
                @ np.diag(wnew)
                @ (State[4:6, :] - avgState[4:6].reshape(2, 1) @ np.ones((1, N))).T
        )
        covDiff = (
                (State[6:8, :] - avgState[6:8].reshape(2, 1) @ np.ones((1, N)))
                @ np.diag(wnew)
                @ (State[6:8, :] - avgState[6:8].reshape(2, 1) @ np.ones((1, N))).T
        )

        e_vals = np.linalg.eigvals(covPos)
        if np.any(e_vals <= 1e-10):
            # print("<e-10")
            covPos += np.eye(covPos.shape[0]) * (1e-10)
            Dpos = np.linalg.cholesky(covPos)

        else:
            Dpos = np.linalg.cholesky(covPos)

        e_vals = np.linalg.eigvals(covQ)
        if np.any(e_vals <= 1e-10):
            # print("<e-10")
            covQ += np.eye(covQ.shape[0]) * (1e-10)
            Dq = np.linalg.cholesky(covQ)
        else:
            Dq = np.linalg.cholesky(covQ)

        e_vals = np.linalg.eigvals(covWind)
        if np.any(e_vals <= 1e-10):
            # print("<e-10")
            covWind += np.eye(covWind.shape[0]) * (1e-10)
            Dwind = np.linalg.cholesky(covWind)

        else:
            Dwind = np.linalg.cholesky(covWind)

        e_vals = np.linalg.eigvals(covDiff)
        if np.any(e_vals <= 1e-10):
            # print("<e-10")
            covDiff += np.eye(covDiff.shape[0]) * (1e-10)
            Ddiff = np.linalg.cholesky(covDiff)

        else:
            Ddiff = np.linalg.cholesky(covDiff)

        wnew, index = resampling_index_np(wnew)
        State = State[:, index]

        # wnew = wnew.T

        A = (4 / (n + 2)) ** (1 / (n + 4))
        hopt = A * (N ** (-1 / (n + 4)))

        idx_ = np.ones(N, dtype=bool)
        newState = State.copy()

        for _ in range(3):
            idx = idx_
            # print(idx)
            newState[:3, idx] = State[:3, idx] + hopt * Dpos @ np.random.randn(
                3, np.sum(idx)
            )
            newState[3, idx] = State[3, idx] + hopt * Dq @ np.random.randn(
                1, np.sum(idx)
            )
            newState[4:6, idx] = State[4:6, idx] + hopt * Dwind @ np.random.randn(
                2, np.sum(idx)
            )
            newState[6:8, idx] = State[6:8, idx] + hopt * Ddiff @ np.random.randn(
                2, np.sum(idx)
            )

            # idx = np.logical_not(gCon(newState))
            idx = np.where(gCon(newState) != 1)[0]
            if np.sum(idx) == 0:
                break
            else:
                newState[:, idx] = State[:, idx]

        newerr = newState - State
        SIG = hopt ** 2 * block_diag(covPos, covQ, covWind, covDiff)

        logratio = -0.5 * np.sum(
            np.dot(newerr.T, np.linalg.inv(SIG)).T * newerr, axis=0
        ) + 0.5 * np.sum(
            np.dot(np.zeros((n, N)).T, np.linalg.inv(SIG)).T * np.zeros((n, N)), axis=0
        )

        xupdate = hLike(State, yobv, pos, hParm, nsg_n)
        xnewupdate = hLike(newState, yobv, pos, hParm, nsg_n)

        alpha = xnewupdate / xupdate * np.exp(logratio)

        mcrand = np.random.rand(N)
        accept = alpha >= mcrand
        reject = alpha < mcrand
        newState[:, reject] = State[:, reject]

        newpart = {
            "x": newState[0, :],
            "y": newState[1, :],
            "z": newState[2, :],
            "Q": newState[3, :],
            "u": newState[4, :],
            "phi": newState[5, :],
            "ci": newState[6, :],
            "cii": newState[7, :],
        }

    else:
        newpart = xpart

    time_taken = time.time() - ct

    xest = None  # Not sure where this is used in your code, so leaving it as None
    info = {
        "ess": ess,
        "avgSampling": 0,  # Not sure where this is calculated, so leaving it as 0
        "time": time_taken,
    }

    return newpart, wnew, info
