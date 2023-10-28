import numpy as np


def fDyn(s, *args):
    np.random.seed(123)
    optargin = len(args)
    N = s["x"].shape[0]

    if optargin == 1:
        sigma = args[0]

        # source location
        splus = {
            "x": s["x"] + sigma["x"] * np.random.randn(N),
            "y": s["y"] + sigma["y"] * np.random.randn(N),
            "z": s["z"] + sigma["z"] * np.random.randn(N),
            "Q": s["Q"] + sigma["Q"] * np.random.randn(N),
            "u": s["u"] + sigma["u"] * np.random.randn(N),
            "phi": s["phi"] + sigma["phi"] * np.random.randn(N),
            "ci": s["ci"] + sigma["ci"] * np.random.randn(N),
            "cii": s["cii"] + sigma["cii"] * np.random.randn(N),
        }

        idx = np.where(splus["Q"] < 0)[0]
        while len(idx) > 0:
            splus["Q"][idx] = s["Q"][idx] + sigma["Q"] * np.random.randn(len(idx))
            idx = np.where(splus["Q"] <= 0)[0]

        idx = np.where(splus["ci"] <= 0)[0]
        while len(idx) > 0:
            splus["ci"][idx] = s["ci"][idx] + sigma["ci"] * np.random.randn(len(idx))
            idx = np.where(splus["ci"] <= 0)[0]

        idx = np.where(splus["cii"] <= 0)[0]
        while len(idx) > 0:
            splus["cii"][idx] = s["cii"][idx] + sigma["cii"] * np.random.randn(len(idx))
            idx = np.where(splus["cii"] <= 0)[0]
    else:
        splus = s

    return splus


def gCon(theta):
    np.random.seed(123)
    if isinstance(theta, dict):
        gVal = np.ones((len(theta["Q"]), 4), dtype=bool)
        gVal[:, 0] = theta["Q"] >= 0
        gVal[:, 1] = theta["u"] >= 0
        gVal[:, 2] = theta["ci"] > 0
        gVal[:, 3] = theta["cii"] > 0

        consTrue = np.prod(gVal, axis=1).astype(int)
    else:
        gVal = np.ones((theta.shape[1], 4), dtype=bool)
        gVal[:, 0] = theta[3, :] >= 0
        gVal[:, 1] = theta[4, :] >= 0
        gVal[:, 2] = theta[6, :] > 0
        gVal[:, 3] = theta[7, :] > 0

        consTrue = np.prod(gVal, axis=1).astype(int)

    return consTrue
