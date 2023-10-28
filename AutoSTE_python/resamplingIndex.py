import numpy as np




def resampling_index(weights, N=None):
    if N is None:

        N = len(weights)

        index = np.random.choice(N, N, p=weights)

    else:

        index = np.random.choice(len(weights), N, p=weights)


    index = np.sort(index)

    wnew = np.ones(N) / N

    return wnew, index




def resampling_index_(weights, N=None):
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
