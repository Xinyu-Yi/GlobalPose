r"""
    Magnetometer calibrator using Figure-8 Calibration.
"""

__all__ = ['MagnetometerCalibrator']


import os
import numpy as np
from scipy.optimize import minimize


class MagnetometerCalibrator:
    r"""
    Magnetometer calibrator using Figure-8 Calibration.
    """
    def __init__(self, state_file='', visualize_per_n_points=-1):
        r"""
        Initialize the magnetometer calibrator.

        :param state_file: Calibration state file.
        :param visualize_per_n_points: Render the point cloud when adding n points. Negative means no visualization.
        """
        self.visualize_per_n_points = int(visualize_per_n_points)
        self.data = np.empty((0, 3))
        self.hard_iron = np.zeros(3)
        self.soft_iron = np.eye(3)

        if state_file != '':
            if os.path.exists(state_file):
                self.load_state(state_file)
            else:
                print('Warning: MagnetometerCalibrator state file {} does not exist.'.format(state_file))

        if self.visualize_per_n_points > 0:
            from articulate.utils.open3d import PointCloud3DViewer
            self.viewer = PointCloud3DViewer()
            self.viewer.connect()

    def add_point(self, mS):
        r"""
        Add magnetic field measurements.

        :param mS: Sensor-frame raw magnetic field measurements in shape [3] or [N, 3].
        """
        mS = np.array(mS).reshape(-1, 3)
        self.data = np.vstack((self.data, mS))
        if self.visualize_per_n_points > 0:
            if len(self.data) // self.visualize_per_n_points > (len(self.data) - len(mS)) // self.visualize_per_n_points:
                self.viewer.update(self.data, reset_view_point=True)

    def calibrate(self):
        r"""
        Perform the magnetic field calibration.
        """
        if len(self.data) < 20:
            raise RuntimeError('Not enough data to calibrate')

        def construct_positive_definite(lower_params):
            L = np.zeros((3, 3))
            L[np.tril_indices(3)] = lower_params
            L[0, 0] = np.exp(L[0, 0])
            L[1, 1] = np.exp(L[1, 1])
            L[2, 2] = np.exp(L[2, 2])
            return L @ L.T

        def objective_function(params):
            hard_iron = params[:3]
            soft_iron = construct_positive_definite(params[3:])
            calibrated_data = np.dot(soft_iron, (self.data - hard_iron).T).T
            magnitude = np.linalg.norm(calibrated_data, axis=1)
            return np.mean((magnitude - 1) ** 2)

        initial_params = np.concatenate([
            np.mean(self.data, axis=0),   # hard iron init
            [np.log(1), 0, 0, np.log(1), 0, np.log(1)]   # soft iron init
        ])
        result = minimize(objective_function, initial_params, method='BFGS')
        self.hard_iron = result.x[:3]
        self.soft_iron = construct_positive_definite(result.x[3:])

    def compute_error(self):
        r"""
        Compute error of the calibration.

        :return: Data coverage and magnitude error.
        """
        calibrated_data = self.apply_calibration(self.data)
        magnitude = np.linalg.norm(calibrated_data, axis=1)
        pts = calibrated_data / magnitude.reshape(-1, 1)
        sph_pts = np.random.randn(1000, 3)
        sph_pts /= np.linalg.norm(sph_pts, axis=1, keepdims=True)
        max_dots = np.max(sph_pts @ pts.T, axis=1)
        coverage = np.mean(np.degrees(np.arccos(max_dots)) <= 5)
        error = np.abs(magnitude - 1).mean()
        return coverage, error

    def apply_calibration(self, mS):
        r"""
        Apply the magnetic field calibration to the raw magnetic field measurements in shape [3] or [N, 3].

        :param mS: Raw magnetic field measurements in shape [3] or [N, 3].
        :return: Calibrated magnetic field measurements in shape [3] or [N, 3].
        """
        raw_data = np.array(mS)
        calibrated_data = np.dot(self.soft_iron, (raw_data.reshape(-1, 3) - self.hard_iron).T).T
        return calibrated_data.reshape(raw_data.shape)

    def save_state(self, filename):
        r"""
        Save the magnetic field calibrator state.
        """
        data = {'data': self.data, 'hard_iron': self.hard_iron, 'soft_iron': self.soft_iron}
        np.savez(filename, **data)

    def load_state(self, filename):
        r"""
        Load the magnetic field calibrator state.
        """
        data = np.load(filename, allow_pickle=True)
        self.data = data['data']
        self.hard_iron = data['hard_iron']
        self.soft_iron = data['soft_iron']

