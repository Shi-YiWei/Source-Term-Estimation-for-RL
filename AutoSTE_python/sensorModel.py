import numpy as np
from plumeModel import plumeModel



# source_true, pos, sensor_model_parameters
def sensorModel(source, pos, sensor_model_parameters):
    # Generate simulated sensor data based on the source term, sensor position, and sensor characteristics

    conc = plumeModel(source, pos)

    # Add noise
    datasize = conc.shape

    error = sensor_model_parameters['sig_pct'] * conc * np.random.randn(*datasize)  # Add noise or fluctuations

    sensorData = conc + error

    # Not detect if below the threshold
    sensorData = np.where(sensorData < sensor_model_parameters['thresh'], 0, sensorData)

    # Not detect due to the mis-detection rate
    mask = np.random.rand(*datasize) < (1 - sensor_model_parameters['Pd'])
    #sensorData = np.where(mask, 0, sensorData)

    return sensorData
