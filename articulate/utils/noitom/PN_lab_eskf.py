r"""
    API for Noitom PN Lab IMU Set using our ESKF implementation.
"""


__all__ = ['IMUSet_ESKF', 'CalibratedIMUSet_ESKF']


from .PN_lab import *
import threading
import torch
import time

class IMUSet_ESKF(IMUSet):
    def __init__(self, eskfs: list, use_6dof=False, udp_port=7777):
        r"""
        Init an IMU set using our own ESKF algorithm.

        :param eskfs: A list of `carticulate.ESKF` filters for all connected IMUs, e.g.,
                      [ESKF(an=5e-2, wn=5e-3, aw=1e-4, ww=1e-5, mn=5e-3) for _ in range(n)]
        :param use_6dof: Whether using 6dof or 9dof ESKF algorithm.
        :param udp_port: Udp port for Axis Lab software.
        """
        super().__init__(udp_port)
        assert len([_ for _ in self.sensors if _ is not None]) == len(eskfs), 'number of eskfs is not equal to number of IMUs'
        self.t = 0
        self.n_imus = len(eskfs)
        self.eskfs = eskfs
        self.using_6dof = use_6dof
        self.thread = threading.Thread(target=self._run)
        self.thread.setDaemon(True)
        self.thread.start()

    def _update(self):
        while True:
            evts = self.app.poll_next_event()
            if len(evts) > 0 and evts[0].timestamp > self.t:
                dt = evts[0].timestamp - self.t
                self.t = evts[0].timestamp
                return dt
            time.sleep(0.001)

    def _get_raw(self):
        a, w, m = [], [], []
        for sensor in self.sensors:
            if sensor is not None:
                a.append(sensor.get_accelerated_velocity())
                w.append(sensor.get_angular_velocity())
                m.append(sensor.get_compass_value())
        wS = torch.tensor(w) / 180 * torch.pi
        mS = torch.tensor(m)
        aS = -torch.tensor(a) / 1000 * self.g  # acceleration is reversed
        return aS, wS, mS

    def _run(self):
        succeed = False
        while not succeed:
            self._update()
            aS, wS, mS = self._get_raw()
            if self.using_6dof:
                succeed = all([self.eskfs[i].initialize_6dof(am=aS[i]) for i in range(self.n_imus)])
            else:
                succeed = all([self.eskfs[i].initialize_9dof(am=aS[i], mm=mS[i]) for i in range(self.n_imus)])
        while True:
            dt = self._update()
            aS, wS, mS = self._get_raw()
            for i in range(self.n_imus):
                self.eskfs[i].predict(am=aS[i], wm=wS[i], dt=dt)
                if self.using_6dof:
                    self.eskfs[i].correct(am=aS[i], wm=wS[i])
                else:
                    self.eskfs[i].correct(am=aS[i], wm=wS[i], mm=mS[i])

    def get(self):
        r"""
        Get the latest data from all sensors (non-block). Check the returned timestamp to see if the data is updated.

        :return: Timestamp in seconds, RIS in [N, 3, 3], aS, wS, mS, aI, wI, mI in [N, 3].
        """
        aS, wS, mS = self._get_raw()
        RIS = torch.stack([torch.from_numpy(eskf.get_orientation_R()).float() for eskf in self.eskfs])
        ab = torch.stack([torch.from_numpy(eskf.get_accelerometer_bias()).float() for eskf in self.eskfs])
        wb = torch.stack([torch.from_numpy(eskf.get_gyroscope_bias()).float() for eskf in self.eskfs])
        aI = RIS.bmm((aS - ab).unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])
        wI = RIS.bmm((wS - wb).unsqueeze(-1)).squeeze(-1)
        mI = RIS.bmm(mS.unsqueeze(-1)).squeeze(-1)
        return self.t, RIS, aS, wS, mS, aI, wI, mI

    def get_noitom(self):
        r"""
        Get the noitom ground-truth (non-block).

        :return: Timestamp in seconds, RIS in [N, 3, 3], aS, wS, mS, aI, wI, mI in [N, 3].
        """
        aS, wS, mS = self._get_raw()
        RIS = self._quaternion_to_rotation_matrix(torch.tensor([sensor.get_posture() for sensor in self.sensors if sensor is not None]))
        aI = RIS.bmm(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])   # calculate global free acceleration
        wI = RIS.bmm(wS.unsqueeze(-1)).squeeze(-1)                     # calculate global angular velocity
        mI = RIS.bmm(mS.unsqueeze(-1)).squeeze(-1)                     # calculate global magnetic field
        return self.t, RIS, aS, wS, mS, aI, wI, mI

class CalibratedIMUSet_ESKF(IMUSet_ESKF):
    _RMB_Npose = CalibratedIMUSet._RMB_Npose
    _RMB_Tpose = CalibratedIMUSet._RMB_Tpose

    def __init__(self, eskfs: list, use_6dof=False, udp_port=7777):
        r"""
        Init a calibrated IMU set using our own ESKF algorithm.

        :param eskfs: A list of `carticulate.ESKF` filters for all connected IMUs, e.g.,
                      [cart.ESKF(an=5e-2, wn=5e-3, aw=1e-4, ww=1e-5, mn=5e-3) for _ in range(n)]
        :param use_6dof: Whether using 6dof or 9dof ESKF algorithm.
        :param udp_port: Udp port for Axis Lab software.
        """
        super().__init__(eskfs, use_6dof, udp_port)
        self.mask = [_ is not None for _ in self.sensors]
        self.N = sum(self.mask)
        self.RMI = torch.eye(3).repeat(self.N, 1, 1)
        self.RSB = torch.eye(3).repeat(self.N, 1, 1)
        self._normalize_tensor = CalibratedIMUSet._normalize_tensor
        self._mean_rotation = CalibratedIMUSet._mean_rotation
        self._rotation_matrix_to_axis_angle = CalibratedIMUSet._rotation_matrix_to_axis_angle
        self._angle_from_two_vectors = CalibratedIMUSet._angle_from_two_vectors
        self._input = CalibratedIMUSet._input
        self._fixpose_calibration = CalibratedIMUSet._fixpose_calibration.__get__(self)
        self._changepose_calibration = CalibratedIMUSet._changepose_calibration.__get__(self)
        self._walking_calibration = CalibratedIMUSet._walking_calibration.__get__(self)
        self.calibrate = CalibratedIMUSet.calibrate.__get__(self)

    def get(self):
        t, RIS, aS, wS, mS, aI, wI, mI = super().get()
        RMB = self.RMI.matmul(RIS).matmul(self.RSB)
        aM = self.RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
        wM = self.RMI.matmul(wI.unsqueeze(-1)).squeeze(-1)
        mM = self.RMI.matmul(mI.unsqueeze(-1)).squeeze(-1)
        return t, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM

    def get_noitom(self):
        t, RIS, aS, wS, mS, aI, wI, mI = super().get_noitom()
        RMB = self.RMI.matmul(RIS).matmul(self.RSB)
        aM = self.RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
        wM = self.RMI.matmul(wI.unsqueeze(-1)).squeeze(-1)
        mM = self.RMI.matmul(mI.unsqueeze(-1)).squeeze(-1)
        return t, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM