r"""
    IMU simulation utils.
"""


import torch


class IMUSimulator:
    r"""
    IMU simulator for a given 6DoF trajectory.
    """
    def __init__(self):
        self.pWS = None
        self.RWS = None
        self.fps = None

    def set_trajectory(self, pWS: torch.Tensor, RWS: torch.Tensor, fps=60):
        r"""
        Set the 6DoF trajectory for the IMU.

        :param pWS: IMU position trajectory in the world frame in shape [..., 3].
        :param RWS: IMU orientation trajectory in the world frame in shape [..., 3, 3].
        :param fps: Frame rate.
        """
        self.pWS = pWS
        self.RWS = RWS
        self.fps = fps

    def get_acceleration(self, gW=(0, -9.8, 0)):
        r"""
        Get the sensor-local acceleration.

        :param gW: Gravity acceleration in the world frame.
        :return: Acceleration in shape [..., 3].
        """
        a = self.pWS[:-2] - 2 * self.pWS[1:-1] + self.pWS[2:]
        a0 = 2 * self.pWS[0] - 5 * self.pWS[1] + 4 * self.pWS[2] - self.pWS[3]
        a1 = 2 * self.pWS[-1] - 5 * self.pWS[-2] + 4 * self.pWS[-3] - self.pWS[-4]
        aW = torch.cat((a0.unsqueeze(0), a, a1.unsqueeze(0)), dim=0) * self.fps * self.fps
        aS = self.RWS.transpose(-1, -2).matmul((aW - torch.tensor(gW, device=aW.device)).unsqueeze(-1)).squeeze(-1)
        return aS

    def get_angular_velocity(self):
        r"""
        Get the sensor-local angular velocity.

        :return: Angular velocity in shape [..., 3].
        """
        Rdot = self.RWS[2:] - self.RWS[:-2]
        Rdot0 = -3 * self.RWS[0] + 4 * self.RWS[1] - self.RWS[2]
        Rdot1 = 3 * self.RWS[-1] - 4 * self.RWS[-2] + self.RWS[-3]
        Rdot = torch.cat((Rdot0.unsqueeze(0), Rdot, Rdot1.unsqueeze(0)), dim=0) * self.fps / 2
        wShat = self.RWS.transpose(-1, -2).matmul(Rdot)
        wShat = (wShat - wShat.transpose(-1, -2)) / 2
        wS = torch.stack((wShat[..., 2, 1], wShat[..., 0, 2], wShat[..., 1, 0]), dim=-1)
        return wS

    def get_magnetic_field(self, mW=(1., 0, 0)):
        r"""
        Get the sensor-local magnetic field.

        :param mW: Magnetic field in the world frame.
        :return: Magnetic field in shape [..., 3].
        """
        mS = self.RWS.transpose(-1, -2).matmul(torch.tensor(mW, device=self.RWS.device).unsqueeze(-1)).squeeze(-1)
        return mS
