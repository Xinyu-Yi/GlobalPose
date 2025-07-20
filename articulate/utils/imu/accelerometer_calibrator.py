r"""
    Accelerometer calibrator.
"""

__all__ = ['AccelerometerCalibrator']


import os
import numpy as np


class AccelerometerCalibrator:
    r"""
    Accelerometer calibrator using two-side method on the three axes.
    """
    g = 9.8

    def __init__(self, state_file=''):
        r"""
        Initialize the accelerometer calibrator.

        :param state_file: Calibration state file.
        """
        self.scale = np.ones(3)
        self.bias = np.zeros(3)

        if state_file != '':
            if os.path.exists(state_file):
                self.load_state(state_file)
            else:
                print('Warning: AccelerometerCalibrator state file {} does not exist.'.format(state_file))

    def calibrate(self, xpos, xneg, ypos, yneg, zpos, zneg):
        r"""
        Calibrate the accelerometer using gravity acceleration measurements.

        :param xpos: Sensor-frame acceleration measurements [g, 0, 0] in shape [N, 3].
        :param xneg: Sensor-frame acceleration measurements [-g, 0, 0] in shape [N, 3].
        :param ypos: Sensor-frame acceleration measurements [0, g, 0] in shape [N, 3].
        :param yneg: Sensor-frame acceleration measurements [0, -g, 0] in shape [N, 3].
        :param zpos: Sensor-frame acceleration measurements [0, 0, g] in shape [N, 3].
        :param zneg: Sensor-frame acceleration measurements [0, 0, -g] in shape [N, 3].
        """
        xpos = np.array(xpos).reshape(-1, 3).mean(axis=0)
        xneg = np.array(xneg).reshape(-1, 3).mean(axis=0)
        ypos = np.array(ypos).reshape(-1, 3).mean(axis=0)
        yneg = np.array(yneg).reshape(-1, 3).mean(axis=0)
        zpos = np.array(zpos).reshape(-1, 3).mean(axis=0)
        zneg = np.array(zneg).reshape(-1, 3).mean(axis=0)
        self.bias[0] = (xpos + xneg)[0] / 2
        self.bias[1] = (ypos + yneg)[1] / 2
        self.bias[2] = (zpos + zneg)[2] / 2
        self.scale[0] = (xpos - xneg)[0] / (2 * self.g)
        self.scale[1] = (ypos - yneg)[1] / (2 * self.g)
        self.scale[2] = (zpos - zneg)[2] / (2 * self.g)

    def apply_calibration(self, aS):
        r"""
        Apply the bias calibration to the raw accelerometer measurements in shape [3] or [N, 3].

        :param aS: Raw accelerometer measurements in shape [3] or [N, 3].
        :return: Calibrated accelerometer measurements in shape [3] or [N, 3].
        """
        return (np.array(aS) - self.bias) / self.scale

    def save_state(self, filename):
        r"""
        Save the accelerometer calibrator state.
        """
        data = {'bias': self.bias, 'scale': self.scale}
        np.savez(filename, **data)

    def load_state(self, filename):
        r"""
        Load the accelerometer calibrator state.
        """
        data = np.load(filename, allow_pickle=True)
        self.bias = data['bias']
        self.scale = data['scale']
