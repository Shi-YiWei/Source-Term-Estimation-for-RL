import numpy as np

from ste.core import (
    PlumeModel,
    SensorModel,
    SourceParam,
    DefaultSourceParam,
    DefaultSensorParam,
    fDyn,
    ConstCheck,
)
from ste.draw import SimpleDrawPlume
from ste.mcmc import (
    MCMCPF,
    LikePlume,
    resamplingIndex,
)


def main():
    s = DefaultSourceParam()
    m = DefaultSensorParam()

    domain = {"xmin": 20, "xmax": 60, "ymin": 40, "ymax": 80, "zmin": 0, "zmax": 10}
    P_k = [2, 2, 4]
    moveDist = 2
    pos = {}
    pos["x_matrix"] = P_k[0]
    pos["y_matrix"] = P_k[1]
    pos["z_matrix"] = P_k[2]

    D = []  # store sensor readings
    N = 20000  # number of particles
    # Uniform prior for location
    theta = SourceParam()
    theta.x = domain.xmin + (domain.xmax - domain.xmin) * np.random.rand(N, 1)
    theta.y = domain.ymin + (domain.ymax - domain.ymin) * np.random.rand(N, 1)
    theta.z = 0 + 5 * np.random.rand(N, 1)

    # Gamma prior for release rate Q
    a = np.ones(N, 1) * 2
    b = np.ones(N, 1) * 5
    theta.Q = np.random.gamma(shape=a, scale=b)
    theta.u = s.u + np.random.randn(N, 1) * 2
    theta.phi = s.phi * 0.9 + np.random.randn(N, 1) * 10.0 * PI / 180
    theta.ci = s.ci + 2 * np.random.rand(N, 1)
    theta.cii = s.cii + 2 * np.random.rand(N, 1) - 2

    Wpnorm = np.ones(N, 1) / N
    f_dyn = fDyn
    g_const = ConstCheck

    for i in range(100):
        # generate sensor data with added noise and miss-detection

        Dsim = SensorModel(s, pos, m)
        theta, Wpnorm, info = MCMCPF(
            k=i,
            xpartminus=theta,
            wminus=Wpnorm,
            yobv=Dsim,
            fDyn=f_dyn,
            fParm=None,
            hLike=LikePlume,
            hParm=m,
            pos=pos,
            gCon=g_const,
            gParm=[],
        )
        # define the action set
        ynew = np.array(
            [
                [0, moveDist, -moveDist, 0],
                2 * [0, moveDist, -moveDist, 0],
                3 * [0, moveDist, -moveDist, 0],
            ]
        )
        xnew = np.array(
            [
                [moveDist, 0, 0, -moveDist],
                2 * [moveDist, 0, 0, -moveDist],
                3 * [moveDist, 0, 0, -moveDist],
            ]
        )
        znew = np.zeros([12, 1])

        Xneighbour = np.zeros(1, xnew.shape[1])
        Yneighbour = np.zeros(1, ynew.shape[1])
        Zneighbour = np.zeros(1, znew.shape[1])
        reward = np.zeros(1, xnew.shape[1])

        # down sample the source term particles (theta_i, i=1,...N) from N to Nz for generating the hypothetical measurements
        Nz = 25
        # the number of hypothetical measurements for each source term particle due to measurement noise
        MM = 1

        _, indx_z = resamplingIndex(Wpnorm, Nz)

        for k in range(xnew.shape[1]):
            Xneighbour[k] = pos.x_matrix + xnew[k]
            Yneighbour[k] = pos.y_matrix + ynew[k]
            Zneighbour[k] = pos.z_matrix + znew[k]

            if Xneighbour[k] < domain.xmin or Xneighbour[k] > domain.xmax or\
                    Yneighbour[k] < domain.ymin or Yneighbour[k] > domain.ymax or\
                    Zneighbour[k] < domain.zmin or Zneighbour[k] > domain.zmax:
                reward[k] = np.nan

            npos = {}
            npos["x_matrix"] = pos.x_matrix + xnew[k]
            npos["y_matrix"] = pos.y_matrix + ynew[k]
            npos["z_matrix"] = pos.z_matrix + znew[k]

            infoGain=0

            for jj in range(Nz):
                d = DefaultSourceParam()
                d.x = theta.x[indx_z[jj]]
                d.y = theta.y[indx_z[jj]]
                d.z = theta.z[indx_z[jj]]
                d.Q = theta.Q[indx_z[jj]]
                d.u = theta.u[indx_z[jj]]
                d.phi = theta.phi[indx_z[jj]]
                d.ci = theta.ci[indx_z[jj]]
                d.cii = theta.cii[indx_z[jj]]




if __name__ == "__main__":
    main()
