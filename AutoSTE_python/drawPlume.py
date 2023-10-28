import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plumeModel import plumeModel
from scipy.constants import pi as PI
import matplotlib.colors as mcolors

def drawPlume(ax, s, domain):
    xmin, xmax, ymin, ymax, zmin, zmax = domain
    nGrid = 200
    height = 0
    x_coord = np.linspace(xmin, xmax, nGrid)
    y_coord = np.linspace(ymin, ymax, nGrid)
    z_coord = np.linspace(zmin, zmax, nGrid)

    X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord)

    ex = {
        'x_matrix': X,
        'y_matrix': Y,
        'z_matrix': Z
    }

    conc = plumeModel(s, ex)

    conc_min, conc_max = np.min(conc[:, :, height]), np.max(conc[:, :, height])
    print(conc_min)
    print(conc_max)
    colors = [(1, 1, 1)] + [plt.cm.jet(i) for i in range(1, 256)]
    newcmp = mcolors.LinearSegmentedColormap.from_list("white_jet", colors, N=256)


    g = ax.pcolor(
        X[:, :, height],
        Y[:, :, height],
        conc[:, :, height],
        cmap=newcmp,
        vmin=conc_min,
        vmax=conc_max,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    # ax.set_aspect("equal")
    plt.colorbar(g, orientation="vertical")

    plt.show()




