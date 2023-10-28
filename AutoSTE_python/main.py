import numpy as np
import matplotlib.pyplot as plt
import drawPlume, sensorModel, fDyn, hLikePlume, gCon, mcmcPF, resamplingIndex, preprocess

np.random.seed(213)

# Simulated source parameters
# true source
source_true = {
    'Q': 5,  # Release rate per time unit
    # source coordinates
    'x': 15.7,
    'y': 16.3,
    'z': 0,
    'u': 2,  # wind speed
    'phi': 45 * np.pi / 180,  # wind direction
    'ci': 2,
    'cii': 10
}

# Create rectangular domain area and draw the plume
xmin = 0
xmax = 25
ymin = 0
ymax = 25
zmin = 0
zmax = 4
domain = [xmin, xmax, ymin, ymax, zmin, zmax]  # Size of search area

# Plot example dispersion from true source

fig1 = plt.figure()
ax1  = plt.axes()
drawPlume.drawPlume(ax1, source_true, domain)


# Sensor model parameters
sensor_model_parameters = {
    'thresh': 5e-3,  # sensor threshold
    'Pd': 0.7,  # probability of detection
    'sig': 1e-4,  # minimum sensor noise
    'sig_pct': 0.5  # the standard deviation is a percentage of the concentration level
}

# Process noise parameters (not used)
sigma = {
    'x': 0.2,
    'y': 0.2,
    'z': 0.1,
    'Q': 0.2,
    'u': 0.2,
    'phi': 2 * np.pi / 180,
    'ci': 0.1,
    'cii': 0.5
}

# Initialization and parameters of the mobile sensor
StartingPosition = np.array([0, 0, 0])  # Starting position [x, y, z]
moveDist = 1  # How far to move for one step

P_k = StartingPosition  # Current robot/sensor position
P_k_store = P_k

pos = {
    'x_matrix': P_k[0],
    'y_matrix': P_k[1],
    'z_matrix': P_k[2]
}

D = []  # Store sensor readings

# Initialize PF
N = 700  # 20000  # number of particles

# Uniform prior for location
theta = {
    'x': xmin + (xmax - xmin) * np.random.rand(N),
    'y': ymin + (ymax - ymin) * np.random.rand(N),
    'z': 0 + 5 * np.random.rand(N),
    'Q': np.random.gamma(2, 5, N),
    'u': source_true['u'] + np.random.randn(N) * 2,
    'phi': source_true['phi'] * 0.9 + np.random.randn(N) * 10 * np.pi / 180,
    'ci': source_true['ci'] + 2 * np.random.rand(N),
    'cii': source_true['cii'] + 2 * np.random.rand(N) - 2
}

w_k_i = np.ones(N) / N

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
preprocess.preprocess(source_true, theta)
plt.show()

for i in range(200):
    print("round:", i)
    # Generate sensor data with added noise and miss-detection

    sensor_data = sensorModel.sensorModel(source_true, pos, sensor_model_parameters)
    # print("Generate sensor data with added noise and miss-detection:", Dsim)

    D.append(sensor_data)

    f_dyn = fDyn.fDyn
    h_likeli = hLikePlume.hLikePlume
    g_const = gCon.gCon

    theta, w_k_i, info = mcmcPF.mcmcPF(theta, w_k_i, sensor_data, f_dyn, sigma, h_likeli, sensor_model_parameters, pos,
                                       None, g_const)

    ''' plot'''

    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(111, projection='3d')
    # # drawPlume.drawPlume(ax3, source_true, domain)
    # S = np.zeros_like(D)
    #
    # for i in range(len(D)):
    #     S[i] = 5 + np.ceil(D[i] * 5e3)
    #
    # ax3.scatter(theta['x'], theta['y'], theta['z'], c='g', marker='o', s=2, alpha=0.03)
    #
    # ax3.plot(pos['x_matrix'], pos['y_matrix'], pos['z_matrix'], 'ro', markerfacecolor='r', markersize=5)
    # ax3.plot(*P_k_store.T, 'r-')
    # ax3.scatter(*P_k_store.T, c='red', marker='o', s=S)
    # ax3.scatter(source_true['x'], source_true['y'], source_true['z'], c='black', marker='s', s=20)
    # ax3.view_init(90, -90)
    # plt.show()


    ''' '''

    # Define the action set
    # xnew = np.array([0, moveDist, -moveDist, 0, 0, 2 * moveDist, -2 * moveDist, 0, 0, 3 * moveDist, -3 * moveDist, 0, ])
    # ynew = np.array([moveDist, 0, 0, -moveDist, 2 * moveDist, 0, 0, -2 * moveDist, 3 * moveDist, 0, 0, -3 * moveDist])
    # znew = np.zeros(12)

    xnew = np.array([0, moveDist, -moveDist, 0])  # , moveDist, -moveDist, moveDist, -moveDist])
    ynew = np.array([moveDist, 0, 0, -moveDist])  # , moveDist, moveDist, -moveDist, -moveDist])
    znew = np.zeros(4)

    Xneighbour = np.zeros(xnew.shape)
    Yneighbour = np.zeros(ynew.shape)
    Zneighbour = np.zeros(znew.shape)

    Nz = 25  # Downsample the source term particles (theta_i, i=1,...N) from N to Nz for generating the hypothetical measurements
    MM = 1  # The number of hypothetical measurements for each source term particle due to measurement noise

    # Downsample the source term particles
    _, indx_z = resamplingIndex.resampling_index(w_k_i, Nz)

    reward = np.zeros(xnew.shape[0])

    # algorithm 2 begin
    for k in range(xnew.shape[0]):
        # print("action:", k)
        Xneighbour[k] = pos['x_matrix'] + xnew[k]
        Yneighbour[k] = pos['y_matrix'] + ynew[k]
        Zneighbour[k] = pos['z_matrix'] + znew[k]

        if (
                pos['x_matrix'] + xnew[k] < xmin or pos['x_matrix'] + xnew[k] > xmax or
                pos['y_matrix'] + ynew[k] < ymin or pos['y_matrix'] + ynew[k] > ymax or
                pos['z_matrix'] + znew[k] < zmin or pos['z_matrix'] + znew[k] > zmax
        ):
            reward[k] = -100  # np.nan
            continue

        new_pos = {
            'x_matrix': pos['x_matrix'] + xnew[k],
            'y_matrix': pos['y_matrix'] + ynew[k],
            'z_matrix': pos['z_matrix'] + znew[k]
        }

        infoGain = 0

        for jj in range(Nz):
            sampled_parameters = {
                'x': theta['x'][indx_z[jj]],
                'y': theta['y'][indx_z[jj]],
                'z': theta['z'][indx_z[jj]],
                'Q': theta['Q'][indx_z[jj]],
                'u': theta['u'][indx_z[jj]],
                'phi': theta['phi'][indx_z[jj]],
                'ci': theta['ci'][indx_z[jj]],
                'cii': theta['cii'][indx_z[jj]]
            }

            for jjj in range(MM):
                z = sensorModel.sensorModel(sampled_parameters, new_pos,
                                            sensor_model_parameters)  # Hypothetical measurements


                # p_z_k_Theta_k =
                # likelihood function
                # w_k_i =  prior distributions from Dirac delta function
                # w_wavy_k1_il = w_k_i * p_z_k_Theta_k # numerator
                # np.sum(w_wavy_k1_il) # denominator
                # w_hat_k1_il # posterior

                p_z_k_Theta_k = hLikePlume.hLikePlume(theta, z, new_pos, sensor_model_parameters) # likelihood function
                w_wavy_k1_il = w_k_i * p_z_k_Theta_k
                w_hat_k1_il = w_wavy_k1_il / np.sum(w_wavy_k1_il)

                WW = w_hat_k1_il / w_k_i
                WW[WW <= 0] = 1
                WW[np.isinf(WW)] = 1
                WW[np.isnan(WW)] = 1

                # Calculate the information gain
                # Comment/uncomment to choose one of those information gains
                # Note: here we used the sum rather than the averaged value

                # ---------------------------------------------------------
                # KLD
                infoGain += np.sum(w_hat_k1_il * np.log(WW))
                # ---------------------------------------------------------

                # ---------------------------------------------------------
                # Entropy
                # infoGain -= np.sum(w_hat_k1_il * np.log2(w_hat_k1_il + (w_hat_k1_il == 0)))
                # ---------------------------------------------------------

                # ---------------------------------------------------------
                # Dual control reward
                #
                # indx = resamplingIndex(w_hat_k1_il, int(N / 5))[1]  # Downsample for quick calculation
                # posPlus = np.array([theta['x'][indx], theta['y'][indx]])
                # posPlus_avg = np.mean(posPlus, axis=1)
                # covPlus = np.cov(posPlus)
                #
                # err_x = posPlus_avg[0] - npos['x_matrix']
                # err_y = posPlus_avg[1] - npos['y_matrix']
                #
                # infoGain -= ((err_x ** 2 + err_y ** 2) + np.trace(covPlus))
                # ---------------------------------------------------------

        reward[k] = infoGain
    #print(reward,'\n')
    # algorithm 2 end
    ind = np.argmax(reward)  # ?
    # print("inforgain:", reward[ind])

    # sorted_indices = np.argsort(reward)
    # ind = sorted_indices[-2]

    pos['x_matrix'] = Xneighbour[ind]
    pos['y_matrix'] = Yneighbour[ind]
    pos['z_matrix'] = Zneighbour[ind]

    P_k = np.array([pos['x_matrix'], pos['y_matrix'], pos['z_matrix']]).T

    P_k_store = np.vstack((P_k_store, P_k))

    # Stop criteria
    _, indx = resamplingIndex.resampling_index(w_k_i)
    Covar = np.cov(np.array([theta['x'][indx], theta['y'][indx]]))
    Spread = np.sqrt(np.trace(Covar))

    if Spread < 0.4:
        print()
        print("round:", i)
        break




_, indx = resamplingIndex.resampling_index(w_k_i)
# print("index:",indx)

theta['x'] = theta['x'][indx]
theta['y'] = theta['y'][indx]
theta['z'] = theta['z'][indx]
theta['Q'] = theta['Q'][indx]
theta['u'] = theta['u'][indx]
theta['phi'] = theta['phi'][indx]
theta['ci'] = theta['ci'][indx]
theta['cii'] = theta['cii'][indx]

estimated_x = np.mean(theta['x'])
estimated_y = np.mean(theta['y'])
print("estimated x", estimated_x)
print("estimated y", estimated_y)



fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
preprocess.preprocess(source_true, theta)
plt.show()
