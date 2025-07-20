r"""
    API for Noitom PN Lab IMU Set.
"""


__all__ = ['IMUSet', 'CalibratedIMUSet']


from .mocap_api import *
import cv2
import numpy as np
import torch
import time
import winsound


class IMUSet:
    g = 9.8

    def __init__(self, udp_port=7777):
        r""" Receiver of Axis Lab software. """
        app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_calc_data()
        app.set_settings(settings)
        app.open()
        time.sleep(0.5)

        sensors = [None for _ in range(6)]
        evts = []
        while len(evts) == 0:
            evts = app.poll_next_event()
            for evt in evts:
                assert evt.event_type == MCPEventType.SensorModulesUpdated
                sensor_module_handle = evt.event_data.sensor_module_data.sensor_module_handle
                sensor_module = MCPSensorModule(sensor_module_handle)
                sensors[sensor_module.get_id() - 1] = sensor_module

        print('find %d sensors' % len([_ for _ in sensors if _ is not None]))
        self.app = app
        self.sensors = sensors
        self.t = 0

    @staticmethod
    def _quaternion_to_rotation_matrix(q: torch.Tensor):
        a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        r = torch.cat((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                       2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                       2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), dim=1)
        return r.view(-1, 3, 3)

    def get(self):
        r"""
        Get the latest data from all sensors (non-block). Check the returned timestamp to see if the data is updated.

        :return: Timestamp in seconds, RIS in [N, 3, 3], aS, wS, mS, aI, wI, mI in [N, 3].
        """
        evts = self.app.poll_next_event()
        if len(evts) > 0:
            self.t = evts[0].timestamp
        q, a, w, m = [], [], [], []
        for sensor in self.sensors:
            if sensor is not None:
                q.append(sensor.get_posture())
                a.append(sensor.get_accelerated_velocity())
                w.append(sensor.get_angular_velocity())
                m.append(sensor.get_compass_value())

        # assuming g is positive (= 9.8), we need to change left-handed system to right-handed by reversing axis x, y, z
        RIS = self._quaternion_to_rotation_matrix(torch.tensor(q))  # rotation is not changed
        wS = torch.tensor(w) / 180 * torch.pi
        mS = torch.tensor(m)
        aS = -torch.tensor(a) / 1000 * self.g                         # acceleration is reversed
        aI = RIS.bmm(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])   # calculate global free acceleration
        wI = RIS.bmm(wS.unsqueeze(-1)).squeeze(-1)                     # calculate global angular velocity
        mI = RIS.bmm(mS.unsqueeze(-1)).squeeze(-1)                     # calculate global magnetic field
        return self.t, RIS, aS, wS, mS, aI, wI, mI

class CalibratedIMUSet(IMUSet):
    _RMB_Npose = torch.tensor([[[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float()
    _RMB_Tpose = torch.eye(3).repeat(6, 1, 1)

    def __init__(self, udp_port=7777):
        super().__init__(udp_port)
        self.mask = [_ is not None for _ in self.sensors]
        self.N = sum(self.mask)
        self.RMI = torch.eye(3).repeat(self.N, 1, 1)
        self.RSB = torch.eye(3).repeat(self.N, 1, 1)

    def get(self):
        t, RIS, aS, wS, mS, aI, wI, mI = super().get()
        RMB = self.RMI.matmul(RIS).matmul(self.RSB)
        aM = self.RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
        wM = self.RMI.matmul(wI.unsqueeze(-1)).squeeze(-1)
        mM = self.RMI.matmul(mI.unsqueeze(-1)).squeeze(-1)
        return t, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM

    @staticmethod
    def _normalize_tensor(x: torch.Tensor, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        normalized_x = x / norm
        return normalized_x

    @staticmethod
    def _mean_rotation(R0, R1):
        R_avg = (R0 + R1) / 2
        U, S, V = torch.svd(R_avg)
        R = torch.matmul(U, V.transpose(-2, -1)).view(-1, 3, 3)
        m = R.det() < 0
        R[m] = U[m].matmul(torch.diag(torch.tensor([1, 1, -1.], device=R.device))).bmm(V[m].transpose(1, 2))
        return R

    @staticmethod
    def _rotation_matrix_to_axis_angle(r):
        result = [cv2.Rodrigues(_)[0] for _ in r.view(-1, 3, 3).numpy()]
        result = torch.from_numpy(np.stack(result)).float().squeeze(-1)
        return result

    @staticmethod
    def _angle_from_two_vectors(v1, v2):
        v1 = CalibratedIMUSet._normalize_tensor(v1.view(-1, 3))
        v2 = CalibratedIMUSet._normalize_tensor(v2.view(-1, 3))
        angle = (v1 * v2).sum(dim=-1).clip(-1, 1).acos() * (180 / np.pi)
        return angle

    @staticmethod
    def _input(s, skip_input):
        if skip_input:
            print(s)
        else:
            return input(s)

    def _fixpose_calibration(self, RMI_method, RSB_method, skip_input=False):
        if RMI_method == '9dof':
            self._input('Put the first imu at pose (x = Right, y = Forward, z = Down, Left-handed) and press enter.', skip_input)
            RSI = self.get()[1][0].t()
            RMI = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).mm(RSI).repeat(self.N, 1, 1)
        elif RMI_method == '6dof':
            self._input('Put all imus at pose (x = Forward, y = Up, z = Left, Left-handed) and press enter.', skip_input)
            RSI = self.get()[1].transpose(1, 2)
            RMI = torch.tensor([[0, 0, -1], [0, -1, 0], [-1, 0, 0.]]).matmul(RSI)
        elif RMI_method == 'skip':
            RMI = self.RMI

        if RSB_method == 'tpose':
            self._input('Stand in T-pose and press enter. The calibration will start in 3 seconds.', skip_input)
            time.sleep(3)
            RIS = self.get()[1]
            RSB = RMI.matmul(RIS).transpose(1, 2).matmul(self._RMB_Tpose[self.mask])
        elif RSB_method == 'npose':
            self._input('Stand in N-pose and press enter. The calibration will start in 3 seconds.', skip_input)
            time.sleep(3)
            RIS = self.get()[1]
            RSB = RMI.matmul(RIS).transpose(1, 2).matmul(self._RMB_Npose[self.mask])

        err = self._angle_from_two_vectors(RMI[:, :, 2], torch.tensor([0, -1, 0.]))
        if all(err < 8) or skip_input:
            c = 'n'
            print('Calibration succeed: My-Iz error %s deg' % err)
        else:
            c = input('Calibration fail: My-Iz error %s deg. Try again? [y]/n' % err)
        if c != 'n':
            self._fixpose_calibration(RMI_method, RSB_method, skip_input)
        else:
            self.RMI = RMI
            self.RSB = RSB

    def _changepose_calibration(self, skip_input=False):
        self._input('Stand in N pose for 3 seconds and then change to T-pose. Press enter to start.', skip_input)
        time.sleep(3)
        RIS_N = self.get()[1]
        winsound.Beep(440, 600)
        print('Change to T-pose now.')
        time.sleep(2)
        RIS_T = self.get()[1]

        yI = torch.tensor([0, 0, -1.])
        zIs = self._rotation_matrix_to_axis_angle(RIS_N[:2].bmm(RIS_T[:2].transpose(1, 2)))
        zIs[0].neg_()
        zIs = self._normalize_tensor(zIs)
        zI = zIs.mean(dim=0)
        xI = self._normalize_tensor(yI.cross(zI, dim=-1))
        zI = self._normalize_tensor(xI.cross(yI, dim=-1))
        RMI = torch.stack([xI, yI, zI], dim=0).repeat(self.N, 1, 1)
        RSB0 = RMI.matmul(RIS_N).transpose(1, 2).matmul(self._RMB_Npose[self.mask])
        RSB1 = RMI.matmul(RIS_T).transpose(1, 2).matmul(self._RMB_Tpose[self.mask])
        RSB = self._mean_rotation(RSB0, RSB1)

        err_forward = self._angle_from_two_vectors(zIs[0], zIs[1])
        err_RSB = self._rotation_matrix_to_axis_angle(RSB0.bmm(RSB1.transpose(1, 2))).norm(dim=-1) * (180 / np.pi)
        if err_forward < 30 and all(err_RSB < 30) or skip_input:
            c = 'n'
            print('Calibration succeed: forward error %.1f deg, RSB error %s deg' % (err_forward, err_RSB))
        else:
            c = input('Calibration fail: forward error %.1f deg, RSB error %s deg. Try again? [y]/n' % (err_forward, err_RSB))
        if c != 'n':
            self._changepose_calibration(skip_input)
        else:
            self.RMI = RMI
            self.RSB = RSB

    def _walking_calibration(self, RMI_method, skip_input=False):
        self._input('Stand in N pose for 3 seconds, then step forward and stop in the N-pose again.', skip_input)
        time.sleep(3)
        RIS_N0 = self.get()[1]
        winsound.Beep(440, 600)
        print('Step forward now.')
        begin_t = last_t = self.get()[0]
        p, v = torch.zeros(self.N, 3), torch.zeros(self.N, 3)
        Cov_pv = Cov_vv = 0
        while last_t - begin_t < 3:
            t, _, aS, wS, _, aI, _, _, _, _, _, _ = self.get()
            if t != last_t:
                dt = t - last_t
                last_t = t
                p += dt * v + 0.5 * dt * dt * aI
                v += dt * aI
                Cov_pv += dt * Cov_vv
                Cov_vv += 1
        p_filtered = p - Cov_pv / Cov_vv * v
        RIS_N1 = self.get()[1]

        if RMI_method == '9dof':
            zI = p_filtered.mean(dim=0).repeat(self.N, 1)
        elif RMI_method == '6dof':
            zI = p_filtered
        yI = torch.tensor([0, 0, -1.]).expand(self.N, 3)
        xI = self._normalize_tensor(yI.cross(zI, dim=-1))
        zI = self._normalize_tensor(xI.cross(yI, dim=-1))
        RMI = torch.stack([xI, yI, zI], dim=-2)
        RSB0 = RMI.matmul(RIS_N0).transpose(1, 2).matmul(self._RMB_Npose[self.mask])
        RSB1 = RMI.matmul(RIS_N1).transpose(1, 2).matmul(self._RMB_Npose[self.mask])
        RSB = self._mean_rotation(RSB0, RSB1)

        # import matplotlib.pyplot as plt
        # plt.scatter([0], [0], label='origin')
        # plt.scatter(p[:, 0], p[:, 1], label='raw')
        # plt.scatter(p_filtered[:, 0], p_filtered[:, 1], label='filtered')
        # plt.legend()
        # plt.show()

        err_vertical = p_filtered[:, -1].abs()
        err_RSB = self._rotation_matrix_to_axis_angle(RSB0.bmm(RSB1.transpose(1, 2))).norm(dim=-1) * (180 / np.pi)
        if all(err_vertical < 0.1) and all(err_RSB < 20) or skip_input:
            c = 'n'
            print('Calibration succeed: vertical error %s m, RSB error %s deg' % (err_vertical, err_RSB))
        else:
            c = input('Calibration fail: vertical error %s m, RSB error %s deg. Try again? [y]/n' % (err_vertical, err_RSB))
        if c != 'n':
            self._walking_calibration(RMI_method, skip_input)
        else:
            self.RMI = RMI
            self.RSB = RSB

    def calibrate(self, method, skip_input=False):
        r"""
        Calibrate the IMU set.

        :param method: Calibration method. Select from:
            - 'tpose_9dof':     Full T-pose calibration for 9-dof IMU. Two steps: align imu 1 and stand in T-pose.
            - 'tpose_6dof':     Full T-pose calibration for 6-dof IMU. Two steps: align all imus and stand in T-pose.
            - 'tpose_onlyRSB':  T-pose calibration only for sensor-to-bone offset. One step: stand in T-pose.
            - 'npose_9dof':     Full N-pose calibration for 9-dof IMU. Two steps: align imu 1 and stand in N-pose.
            - 'npose_6dof':     Full N-pose calibration for 6-dof IMU. Two steps: align all imus and stand in N-pose.
            - 'npose_onlyRSB':  N-pose calibration only for sensor-to-bone offset. One step: stand in N-pose.
            - 'npose_tpose':    Change-pose calibration for 9-dof IMU. One step: stand in N pose and then change to T-pose.
            - 'walking_9dof':   Walking calibration for 9-dof IMU. One step: stand in N pose, step forward and stop in N pose.
            - 'walking_6dof':   Walking calibration for 6-dof IMU. One step: stand in N pose, step forward and stop in N pose.
        :param skip_input: If true, skip user input and do calibration directly without error check.
        """
        if method == 'tpose_9dof':
            self._fixpose_calibration(RMI_method='9dof', RSB_method='tpose', skip_input=skip_input)
        elif method == 'tpose_6dof':
            self._fixpose_calibration(RMI_method='6dof', RSB_method='tpose', skip_input=skip_input)
        elif method == 'tpose_onlyRSB':
            self._fixpose_calibration(RMI_method='skip', RSB_method='tpose', skip_input=skip_input)
        elif method == 'npose_9dof':
            self._fixpose_calibration(RMI_method='9dof', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_6dof':
            self._fixpose_calibration(RMI_method='6dof', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_onlyRSB':
            self._fixpose_calibration(RMI_method='skip', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_tpose':
            self._changepose_calibration(skip_input=skip_input)
        elif method == 'walking_9dof':
            self._walking_calibration(RMI_method='9dof', skip_input=skip_input)
        elif method == 'walking_6dof':
            self._walking_calibration(RMI_method='6dof', skip_input=skip_input)
