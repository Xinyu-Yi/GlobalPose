"""
articulate utils in C++
"""
from __future__ import annotations
import numpy
import typing
__all__ = ['DynamicArmature', 'DynamicModel', 'ESKF', 'KinematicArmature', 'KinematicModel', 'KinematicOptimizer', 'Observation', 'OrientationObservation', 'Position2DObservation', 'Position3DObservation', 'RobustKernel']
M = typing.TypeVar("M", bound=int)
N = typing.TypeVar("N", bound=int)
class DynamicArmature:
    @typing.overload
    def __init__(self) -> None:
        """
        initialize an empty dynamic armature
        """
    @typing.overload
    def __init__(self, armature_file: str) -> None:
        """
        initialize a dynamic armature from file
        """
    def print(self) -> None:
        """
        print armature information
        """
    @property
    def bone(self) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        joint local position expressed in the parent frame
        """
    @bone.setter
    def bone(self, arg1: numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]) -> None:
        ...
    @property
    def com(self) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        body center of mass in the joint frame
        """
    @com.setter
    def com(self, arg1: numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]) -> None:
        ...
    @property
    def gravity(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        gravitional acceleration in the world frame
        """
    @gravity.setter
    def gravity(self, arg0: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
        ...
    @property
    def inertia(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        body inertia in the joint frame (Ixx, Iyy, Izz, Ixy, Iyz, Ixz)
        """
    @inertia.setter
    def inertia(self, arg1: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> None:
        ...
    @property
    def mass(self) -> list[float]:
        """
        body mass
        """
    @mass.setter
    def mass(self, arg0: list[float]) -> None:
        ...
    @property
    def n_joints(self) -> int:
        """
        number of joints
        """
    @n_joints.setter
    def n_joints(self, arg0: int) -> None:
        ...
    @property
    def name(self) -> str:
        """
        armature name
        """
    @name.setter
    def name(self, arg0: str) -> None:
        ...
    @property
    def parent(self) -> list[int]:
        """
        parent joint index (must satisfying parent[i] < i)
        """
    @parent.setter
    def parent(self, arg0: list[int]) -> None:
        ...
class DynamicModel:
    class ExternalForce:
        @typing.overload
        def __init__(self) -> None:
            """
            initialize an empty enternal force
            """
        @typing.overload
        def __init__(self, joint_idx: int, force: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
            """
            initialize an enternal force
            """
        @property
        def force(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
            """
            force applied to the joint in the world frame
            """
        @force.setter
        def force(self, arg0: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
            ...
        @property
        def joint_idx(self) -> int:
            """
            joint index
            """
        @joint_idx.setter
        def joint_idx(self, arg0: int) -> None:
            ...
        @property
        def local_position(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
            """
            position of the force in the joint frame
            """
        @local_position.setter
        def local_position(self, arg0: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
            ...
    class ExternalTorque:
        @typing.overload
        def __init__(self) -> None:
            """
            initialize an empty enternal torque
            """
        @typing.overload
        def __init__(self, joint_idx: int, torque: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
            """
            initialize an enternal torque
            """
        @property
        def joint_idx(self) -> int:
            """
            joint index
            """
        @joint_idx.setter
        def joint_idx(self, arg0: int) -> None:
            ...
        @property
        def torque(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
            """
            torque applied to the joint in the world frame
            """
        @torque.setter
        def torque(self, arg0: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
            ...
    @typing.overload
    def __init__(self, armature_file: str) -> None:
        """
        initialize a dynamic model from armature file
        """
    @typing.overload
    def __init__(self, armature: DynamicArmature) -> None:
        """
        initialize a dynamic model from armature
        """
    def forward_dynamics(self, force: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]], external_force: list[DynamicModel.ExternalForce] = [], external_torque: list[DynamicModel.ExternalTorque] = []) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        compute acceleration given generalized force and external force & torque
        """
    def get_angular_Jacobian(self, joint_idx: int) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        get angular Jacobian: world-frame angular velocity = J * vel
        """
    def get_angular_Jacobian_dot(self, joint_idx: int) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        get the time derivate of angular Jacobian
        """
    def get_angular_velocity(self, joint_idx: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get angular velocity in the world frame
        """
    def get_armature(self) -> DynamicArmature:
        """
        get the armature
        """
    def get_linear_Jacobian(self, joint_idx: int, local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        get linear Jacobian: world-frame linear velocity = J * vel
        """
    def get_linear_Jacobian_dot(self, joint_idx: int, local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        get the time derivate of linear Jacobian
        """
    def get_linear_velocity(self, joint_idx: int, local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get linear velocity in the world frame
        """
    def get_orientation_R(self, joint_idx: int, local_orientation: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[3]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[3]], numpy.dtype[numpy.float32]]:
        """
        get orientation in the world frame (rotation matrix)
        """
    def get_orientation_q(self, joint_idx: int, local_orientation: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get orientation in the world frame (quaternion)
        """
    def get_position(self, joint_idx: int, local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get position in the world frame
        """
    def get_state_R(self) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]]]:
        """
        get the pose, translation, and velocity (rotation matrix)
        """
    def get_state_q(self) -> tuple[numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]], numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]]]:
        """
        get the pose, translation, and velocity (quaternion)
        """
    def inverse_dynamics(self, acc: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]], external_force: list[DynamicModel.ExternalForce] = [], external_torque: list[DynamicModel.ExternalTorque] = []) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        compute generalized force given acceleration and external force & torque
        """
    def mass_matrix(self) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        compute mass matrix
        """
    def print(self) -> None:
        """
        print model information
        """
    def set_state_R(self, pose: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], tran: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], vel: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
        """
        set the pose, translation, and velocity (rotation matrix)
        """
    def set_state_q(self, pose: numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]], tran: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], vel: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
        """
        set the pose, translation, and velocity (quaternion)
        """
    def update_state(self, acc: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]], delta_t: float) -> None:
        """
        update the pose, translation, and velocity by acceleration
        """
class ESKF:
    null_observation: typing.ClassVar[numpy.ndarray]  # value = array([2023.,    6.,   10.], dtype=float32)
    def __init__(self, an: float, wn: float, aw: float, ww: float, mn: float) -> None:
        """
        initialize ESKF with accelerometer[a]/gyroscope[w]/magnetometer[m] measurement noise[n]/random walk[w] standard deviation
        """
    def correct(self, am: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ..., wm: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ..., mm: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ..., pm: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ..., vm: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ..., pn: float = 0.01, vn: float = 0.01) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        ESKF correction with accelerometer[a]/gyroscope[w]/magnetometer[m]/global position[p]/global velocity[v] measurement[m]/noise standard deviation[n], and returns observation scores for debug
        """
    def get_accelerometer_bias(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get accelerometer bias estimation
        """
    def get_gravity_vector(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get gravity vector
        """
    def get_gyroscope_bias(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get gyroscope bias estimation
        """
    def get_magnetic_field_vector(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get magnetic field vector
        """
    def get_orientation_R(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[3]], numpy.dtype[numpy.float32]]:
        """
        get sensor orientation estimation (rotation matrix)
        """
    def get_orientation_q(self) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get sensor orientation estimation (quaternion)
        """
    def get_position(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get sensor position estimation
        """
    def get_state_covariance_matrix(self) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        get state covariance matrix
        """
    def get_velocity(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get sensor velocity estimation
        """
    @typing.overload
    def initialize_6dof(self, RIS: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[3]], numpy.dtype[numpy.float32]], gI: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> bool:
        """
        6dof initialization with known sensor orientation RIS and gravity vector gI, and return succeed or not
        """
    @typing.overload
    def initialize_6dof(self, am: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> bool:
        """
        6dof initialization with accelerometer measurement am and return succeed or not
        """
    @typing.overload
    def initialize_9dof(self, RIS: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[3]], numpy.dtype[numpy.float32]], gI: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], nI: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> bool:
        """
        9dof initialization with known sensor orientation RIS, gravity vector gI, magnetic field vector nI, and return succeed or not
        """
    @typing.overload
    def initialize_9dof(self, am: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], mm: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> bool:
        """
        9dof initialization with accelerometer measurement am and magnetometer measurement mm, and return succeed or not
        """
    def predict(self, am: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], wm: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], dt: float) -> None:
        """
        ESKF prediction with accelerometer[a]/gyroscope[w] measurement[m] and time interval dt
        """
class KinematicArmature:
    @typing.overload
    def __init__(self) -> None:
        """
        initialize an empty kinematic armature
        """
    @typing.overload
    def __init__(self, armature_file: str) -> None:
        """
        initialize a kinematic armature from file
        """
    def print(self) -> None:
        """
        print armature information
        """
    @property
    def bone(self) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        joint local position expressed in the parent frame
        """
    @bone.setter
    def bone(self, arg1: numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]) -> None:
        ...
    @property
    def n_joints(self) -> int:
        """
        number of joints
        """
    @n_joints.setter
    def n_joints(self, arg0: int) -> None:
        ...
    @property
    def name(self) -> str:
        """
        armature name
        """
    @name.setter
    def name(self, arg0: str) -> None:
        ...
    @property
    def parent(self) -> list[int]:
        """
        parent joint index (must satisfying parent[i] < i)
        """
    @parent.setter
    def parent(self, arg0: list[int]) -> None:
        ...
class KinematicModel:
    @typing.overload
    def __init__(self, armature_file: str) -> None:
        """
        initialize a kinematic model from armature file
        """
    @typing.overload
    def __init__(self, armature: KinematicArmature) -> None:
        """
        initialize a kinematic model from armature
        """
    def get_armature(self) -> KinematicArmature:
        """
        get the armature
        """
    def get_orientation_Jacobian_R(self, joint_idx: int, local_orientation: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[3]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        get orientation (rotation matrix) Jacobian: R(state + delta) = R(state) + J * delta. R is flatten to 9x1 by concatenating three column vectors.
        """
    def get_orientation_Jacobian_q(self, joint_idx: int, local_orientation: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        get orientation (quaternion) Jacobian: q(state + delta) = q(state) + J * delta. q is 4x1 in (w, x, y, z) order in the Jacobian.
        """
    def get_orientation_R(self, joint_idx: int, local_orientation: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[3]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[3]], numpy.dtype[numpy.float32]]:
        """
        get orientation in the world frame (rotation matrix)
        """
    def get_orientation_q(self, joint_idx: int, local_orientation: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get orientation in the world frame (quaternion)
        """
    def get_position(self, joint_idx: int, local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]:
        """
        get position in the world frame
        """
    def get_position_Jacobian(self, joint_idx: int, local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]] = ...) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]]:
        """
        get position Jacobian: p(state + delta) = p(state) + J * delta
        """
    def get_state_R(self) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]]:
        """
        get the pose and translation (rotation matrix)
        """
    def get_state_q(self) -> tuple[numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]], numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]]:
        """
        get the pose and translation (quaternion)
        """
    def print(self) -> None:
        """
        print model information
        """
    def set_state_R(self, pose: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], tran: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
        """
        set the pose and translation (rotation matrix)
        """
    def set_state_q(self, pose: numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float32]], tran: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
        """
        set the pose and translation (quaternion)
        """
    def update_state(self, delta: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float32]]) -> None:
        """
        update the pose and translation (right pertubation, translation first)
        """
class KinematicOptimizer:
    def __init__(self, model: KinematicModel, manage_observations: bool = False, verbose: bool = True) -> None:
        """
        initialize a kinematic optimizer  (please set manage_observations to false as they are managed by pybind)
        """
    def add_observation(self, obs: Observation) -> None:
        """
        add observation
        """
    def clear_observations(self) -> None:
        """
        clear all observations
        """
    def get_model(self) -> KinematicModel:
        """
        get the kinematic model
        """
    def optimize(self, iterations: int, init_lambda: float = -1) -> None:
        """
        optimize the state of the kinematic model
        """
    def print(self) -> None:
        """
        print observation information
        """
    def set_constraints(self, optimize_pose: list[bool], optimize_tran: bool) -> None:
        """
        set constraints
        """
class Observation:
    pass
class OrientationObservation(Observation):
    def __init__(self, joint_idx: int, local_orientation: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[1]], numpy.dtype[numpy.float32]], observation: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[1]], numpy.dtype[numpy.float32]], robust_kernel: RobustKernel = RobustKernel.NONE, robust_kernel_delta: float = 1, weight: float = 1) -> None:
        """
        initialize an orientation observation
        """
    def print(self) -> None:
        """
        print observation information
        """
class Position2DObservation(Observation):
    def __init__(self, joint_idx: int, local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], observation: numpy.ndarray[tuple[typing.Literal[2], typing.Literal[1]], numpy.dtype[numpy.float32]], KT: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[4]], numpy.dtype[numpy.float32]], robust_kernel: RobustKernel = RobustKernel.NONE, robust_kernel_delta: float = 1, weight: float = 1) -> None:
        """
        initialize a position 2D observation
        """
    def print(self) -> None:
        """
        print observation information
        """
class Position3DObservation(Observation):
    def __init__(self, joint_idx: int, local_position: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], observation: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float32]], robust_kernel: RobustKernel = RobustKernel.NONE, robust_kernel_delta: float = 1, weight: float = 1) -> None:
        """
        initialize a position 3D observation
        """
    def print(self) -> None:
        """
        print observation information
        """
class RobustKernel:
    """
    Members:
    
      NONE
    
      HUBER
    
      CAUCHY
    
      TUKEY
    """
    CAUCHY: typing.ClassVar[RobustKernel]  # value = <RobustKernel.CAUCHY: 2>
    HUBER: typing.ClassVar[RobustKernel]  # value = <RobustKernel.HUBER: 1>
    NONE: typing.ClassVar[RobustKernel]  # value = <RobustKernel.NONE: 0>
    TUKEY: typing.ClassVar[RobustKernel]  # value = <RobustKernel.TUKEY: 3>
    __members__: typing.ClassVar[dict[str, RobustKernel]]  # value = {'NONE': <RobustKernel.NONE: 0>, 'HUBER': <RobustKernel.HUBER: 1>, 'CAUCHY': <RobustKernel.CAUCHY: 2>, 'TUKEY': <RobustKernel.TUKEY: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
