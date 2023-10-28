import numpy as np
import matplotlib.pyplot as plt

from .core import plumeModel


def drawPlume2D(ax, s, domain, nGrid=200, height=0):
    """
    Draw the plume in 2D
    """
    xmin, xmax, ymin, ymax, zmin, zmax = domain

    x_coord = np.linspace(xmin, xmax, nGrid)
    y_coord = np.linspace(ymin, ymax, nGrid)
    z_coord = np.linspace(zmin, zmax, nGrid)
    X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord)

    ex = {"x_matrix": X, "y_matrix": Y, "z_matrix": Z}

    conc = plumeModel(s, ex)

    conc_min, conc_max = np.min(conc[:, :, height]), np.max(conc[:, :, height])
    g = ax.pcolor(
        X[:, :, height],
        Y[:, :, height],
        conc[:, :, height],
        cmap="PuRd",
        vmin=conc_min,
        vmax=conc_max,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(g, orientation="vertical")
    plt.draw()

    return g


def drawPlume(ax, s, domain,nsg, nGrid=100):
    xmin, xmax, ymin, ymax, zmin, zmax = domain

    x_coord = np.linspace(xmin, xmax, nGrid)
    y_coord = np.linspace(ymin, ymax, nGrid)
    z_coord = np.linspace(zmin, zmax, nGrid)

    X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord)

    ex = {"x_matrix": X, "y_matrix": Y, "z_matrix": Z}

    conc = plumeModel(s, ex, nsg)

    g = ax.plot_surface(
        ex["x_matrix"][:, :, 0],
        ex["y_matrix"][:, :, 0],
        ex["z_matrix"][:, :, 0],
        rstride=1,
        cstride=1,
        facecolors=plt.cm.jet(conc[:, :, 0]),
        alpha=0.1,
        linewidth=0,
        antialiased=False,

    )

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(30, -120)

    ax.scatter(s["x"], s["y"], s["z"], c="k", marker=".", s=20)
    plt.draw()

    return g


def preprocess(s, t):
    histograms = []
    mean_value = []
    _, axs = plt.subplots(nrows=8, ncols=1)

    i = 0
    for key, data in t.items():
        ax = axs[i]
        counts, bins = np.histogram(data)

        mean_value.append(np.mean(data))
        if key == "phi":
            ax.plot([s[key] * 180 / np.pi, s[key] * 180 / np.pi], [0, 1], "r")
        else:
            ax.plot([s[key], s[key]], [0, 1], "r")
        ax.hist(bins[:-1], bins, weights=counts, density=True)
        plt.xlabel(key)
        histograms.append((counts, bins))
        i += 1

    plt.draw()
    plt.show()

    print(mean_value[0], mean_value[1], mean_value[2])
    return histograms
