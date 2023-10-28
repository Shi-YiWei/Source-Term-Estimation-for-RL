import matplotlib.pyplot as plt
import numpy as np


def preprocess(source, theta):
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    i = 0
    for key, data in theta.items():

        ax = axs[i // 2, i % 2]

        ax.axvline(source[key], linewidth=4, color='r')

        # if key == "phi":
        #     ax.plot([source[key] * 180 / np.pi, source[key] * 180 / np.pi], [0, 1], "r")
        # else:
        #     ax.plot([source[key], source[key]], [0, 1], "r")
        if key == 'phi':
            ax.set_xlabel(r'$\phi$')
        else:
            ax.set_xlabel(key)

        ax.hist(data, bins=20, edgecolor='black')
        plt.xlabel(key)

        i += 1

    plt.draw()
    plt.show()

# def preprocess(source, theta):
#     fig, axs = plt.subplots(4, 2, figsize=(15, 15))
#     histograms = []
#
#     i = 0
#     for key, data in theta.items():
#
#         ax = axs[i // 2, i % 2]
#         counts, bins = np.histogram(data)
#
#
#         if key == "phi":
#             ax.plot([source[key] * 180 / np.pi, source[key] * 180 / np.pi], [0, 1], "r")
#         else:
#             ax.plot([source[key], source[key]], [0, 1], "r")
#
#         ax.hist(bins[:-1], bins, weights=counts, density=True)
#         plt.xlabel(key)
#         histograms.append((counts, bins))
#         i += 1
#
#     plt.draw()
#     return histograms
