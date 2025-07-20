r"""
    View human motions in real-time using Unity3D. This is the server script.
"""

__all__ = ['MotionViewer']


import time
import numpy as np
import matplotlib
import socket
import cv2


class MotionViewer:
    r"""
    View human motions in real-time / offline using Unity3D.
    """
    colors = matplotlib.colormaps['tab10'].colors
    ip = '127.0.0.1'
    port = 8888

    def __init__(self, n=1, overlap=True, names=None):
        r"""
        :param n: Number of human motions to simultaneously show.
        :param names: List of str. Subject names. No special char like #, !, @, $.
        """
        assert n <= len(self.colors), 'Subjects are more than colors in MotionViewer.'
        assert names is None or n <= len(names), 'Subjects are more than names in MotionViewer.'
        self.n = n
        self.offsets = [(((n - 1) / 2 - i) * 1.2 if not overlap else 0, 0, 0) for i in range(n)]
        self.names = names
        self.conn = None
        self.server_for_unity = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        self.server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_for_unity.bind((self.ip, self.port))
        self.server_for_unity.listen(1)
        print('MotionViewer server start at', (self.ip, self.port), '. Waiting for unity3d to connect.')
        self.conn, addr = self.server_for_unity.accept()
        s = str(self.n) + '#' + \
            ','.join(['%g' % v for v in np.array(self.colors)[:self.n].ravel()]) + '#' + \
            (','.join(self.names) if self.names is not None else '') + '$'
        self.conn.send(s.encode('utf8'))
        assert self.conn.recv(32).decode('utf8') == '1', 'MotionViewer failed to connect to unity.'
        print('MotionViewer connected to', addr)

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if self.conn is not None:
            self.conn.shutdown(socket.SHUT_RDWR)
            self.conn.close()
        if self.server_for_unity is not None:
            self.server_for_unity.close()
        self.conn = None
        self.server_for_unity = None

    def update_all(self, poses: list, trans: list, render=True):
        r"""
        Update all subject's motions together.

        :param poses: List of pose tensor/ndarray that can all reshape to [24, 3, 3].
        :param trans: List of tran tensor/ndarray that can all reshape to [3].
        :param render: Render the frame after all subjects have been updated.
        """
        assert len(poses) == len(trans) == self.n, 'Number of motions is not equal to the init value in MotionViewer.'
        for i, (pose, tran) in enumerate(zip(poses, trans)):
            self.update(pose, tran, i, render=False)
        if render:
            self.render()

    def update(self, pose, tran, index=0, render=True):
        r"""
        Update the ith subject's motion using smpl pose and tran.

        :param pose: Tensor or ndarray that can reshape to [24, 3, 3] for smpl pose.
        :param tran: Tensor or ndarray that can reshape to [3] for smpl tran.
        :param index: The index of the subject to update.
        :param render: Render the frame after the subject has been updated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        pose = np.array(pose).reshape((24, 3, 3))
        tran = np.array(tran).reshape(3) + np.array(self.offsets[index])
        pose = np.stack([cv2.Rodrigues(_)[0] for _ in pose])
        s = str(index) + '#' + \
            ','.join(['%g' % v for v in pose.ravel()]) + '#' + \
            ','.join(['%g' % v for v in tran.ravel()]) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def draw_plane(self, position, normal, color=(0, 0, 0), size=1.0, render=True):
        r"""
        Draw a plane.

        :param position: Tensor or ndarray that can reshape to [3] for the position.
        :param normal: Tensor or ndarray that can reshape to [3] for the normal.
        :param color: Tensor or ndarray that can reshape to [3] for RGB or [4] for RGBA in [0, 1].
        :param size: Plane size.
        :param render: Render the frame after the plane has been drawn.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        position = np.array(position)
        normal = np.array(normal)
        color = np.array(color)
        s = 'F#' + \
            ','.join(['%g' % v for v in position.ravel()]) + '#' + \
            ','.join(['%g' % v for v in normal.ravel()]) + '#' + \
            ','.join(['%g' % v for v in color.ravel()]) + '#' + \
            str(size) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def draw_plane_from_joint(self, subject_index, joint_index, normal, color=(0, 0, 0), size=1.0, render=True):
        r"""
        Draw a plane from a joint.

        :param subject_index: Subject index to determine the joint.
        :param joint_index: Joint index for the plane position.
        :param normal: Tensor or ndarray that can reshape to [3] for the normal.
        :param color: Tensor or ndarray that can reshape to [3] for RGB or [4] for RGBA in [0, 1].
        :param size: Plane size.
        :param render: Render the frame after the plane has been drawn.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        normal = np.array(normal)
        color = np.array(color)
        s = 'G#' + \
            str(subject_index) + '#' + \
            str(joint_index) + '#' + \
            ','.join(['%g' % v for v in normal.ravel()]) + '#' + \
            ','.join(['%g' % v for v in color.ravel()]) + '#' + \
            str(size) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def clear_plane(self, render=True):
        r"""
        Clear all planes.

        :param render: Render the frame after the line has been cleared.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        self.conn.send('f$'.encode('utf8'))
        if render:
            self.render()

    def draw_line(self, start, end, color=(0, 0, 0), width=0.01, render=True):
        r"""
        Draw a line.

        :param start: Tensor or ndarray that can reshape to [3] for the starting point.
        :param end: Tensor or ndarray that can reshape to [3] for the ending point.
        :param color: Tensor or ndarray that can reshape to [3] for RGB or [4] for RGBA in [0, 1].
        :param width: Line width.
        :param render: Render the frame after the line has been drawn.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        start = np.array(start)
        end = np.array(end)
        color = np.array(color)
        s = 'L#' + \
            ','.join(['%g' % v for v in start.ravel()]) + '#' + \
            ','.join(['%g' % v for v in end.ravel()]) + '#' + \
            ','.join(['%g' % v for v in color.ravel()]) + '#' + \
            str(width) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def draw_line_from_joint(self, subject_index, joint_index, ray, color=(0, 0, 0), width=0.01, render=True):
        r"""
        Draw a line from a joint.

        :param subject_index: Subject index to determine the joint.
        :param joint_index: Joint index for the starting point.
        :param ray: Tensor or ndarray that can reshape to [3] for the line direction * length.
        :param color: Tensor or ndarray that can reshape to [3] for RGB or [4] for RGBA in [0, 1].
        :param width: Line width.
        :param render: Render the frame after the line has been drawn.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        ray = np.array(ray)
        color = np.array(color)
        s = 'R#' + \
            str(subject_index) + '#' + \
            str(joint_index) + '#' + \
            ','.join(['%g' % v for v in ray.ravel()]) + '#' + \
            ','.join(['%g' % v for v in color.ravel()]) + '#' + \
            str(width) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def clear_line(self, render=True):
        r"""
        Clear all lines.

        :param render: Render the frame after the line has been cleared.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        self.conn.send('l$'.encode('utf8'))
        if render:
            self.render()

    def draw_point(self, position, color=(0, 0, 0), radius=0.2, render=True):
        r"""
        Draw a point.

        :param position: Tensor or ndarray that can reshape to [3] for the point position.
        :param color: Tensor or ndarray that can reshape to [3] for RGB or [4] for RGBA in [0, 1].
        :param radius: Point size.
        :param render: Render the frame after the line has been drawn.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        position = np.array(position)
        color = np.array(color)
        s = 'P#' + \
            ','.join(['%g' % v for v in position.ravel()]) + '#' + \
            ','.join(['%g' % v for v in color.ravel()]) + '#' + \
            str(radius) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def draw_point_from_joint(self, subject_index, joint_index, color=(0, 0, 0), radius=0.2, render=True):
        r"""
        Draw a point from a joint.

        :param subject_index: Subject index to determine the joint.
        :param joint_index: Joint index for the point.
        :param color: Tensor or ndarray that can reshape to [3] for RGB or [4] for RGBA in [0, 1].
        :param radius: Point size.
        :param render: Render the frame after the line has been drawn.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        color = np.array(color)
        s = 'J#' + \
            str(subject_index) + '#' + \
            str(joint_index) + '#' + \
            ','.join(['%g' % v for v in color.ravel()]) + '#' + \
            str(radius) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def clear_point(self, render=True):
        r"""
        Clear all points.

        :param render: Render the frame after the line has been cleared.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        self.conn.send('p$'.encode('utf8'))
        if render:
            self.render()

    def update_torque_all(self, torques: list, render=True):
        r"""
        Update all subject's torques together.

        :param torques: List of torque tensor/ndarray that can all reshape to [24, 3] in the joint local frame.
        :param render: Render the frame after all subjects have been updated.
        """
        assert len(torques) == self.n, 'Number of torques is not equal to the init value in MotionViewer.'
        for i, torque in enumerate(torques):
            self.update_torque(torque, i, render=False)
        if render:
            self.render()

    def update_torque(self, torque, index=0, render=True):
        r"""
        Update the ith subject's joint torque.

        :param torque: Tensor or ndarray that can reshape to [24, 3] for joint torque in the local frame.
        :param index: The index of the subject to update.
        :param render: Render the frame after the subject has been updated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        torque = np.array(torque).reshape((24, 3))
        s = 'T#' + \
            str(index) + '#' + \
            ','.join(['%g' % v for v in torque.ravel()]) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def hide_torque(self, subject_index=0, render=True):
        r"""
        Hide all joint torques of a character.

        :param subject_index: Subject index.
        :param render: Render the frame after the subject has been updated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        s = 't#' + str(subject_index) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def show_torque(self, subject_index=0, joint_mask=None, render=True):
        r"""
        Show specific joint torques of a character.

        :param subject_index: Subject index.
        :param joint_mask: List of joint index to show the torque.
        :param render: Render the frame after the subject has been updated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        if joint_mask is None:
            joint_mask = list(range(24))
        s = 'M#' + str(subject_index) + '#' + ','.join(['%d' % v for v in joint_mask]) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def hide_character(self, subject_index=0, render=True):
        r"""
        Hide a character.

        :param subject_index: Subject index.
        :param render: Render the frame after the subject has been updated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        s = 'H#' + str(subject_index) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def show_character(self, subject_index=0, render=True):
        r"""
        Show a character.

        :param subject_index: Subject index.
        :param render: Render the frame after the subject has been updated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        s = 'h#' + str(subject_index) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def instantiate(self, prefab_index, name, position=(0, 0, 0), render=True):
        r"""
        Instantiate a prefab.

        :param prefab_index: Prefab index.
        :param name: Name of the instance.
        :param position: Tensor or ndarray that can reshape to [3] for the prefab position.
        :param render: Render the frame after the prefab has been instantiated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        assert all(c.isalnum() or c == '_' for c in str(name)), 'name should only contains numbers, letters, or _.'
        position = np.array(position)
        s = 'I#' + \
            str(prefab_index) + '#' + \
            str(name) + '#' + \
            ','.join(['%g' % v for v in position.ravel()]) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def destroy(self, name, render=True):
        r"""
        Destroy a prefab instance.

        :param name: Name of the instance.
        :param render: Render the frame after the prefab has been instantiated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        s = 'i#' + \
            str(name) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def render(self):
        r"""
        Render the frame in unity.
        """
        self.conn.send('!$'.encode('utf8'))

    def view_offline(self, poses: list, trans: list, fps=60):
        r"""
        View motion sequences offline.

        :param poses: List of pose tensor/ndarray that can all reshape to [N, 24, 3, 3].
        :param trans: List of tran tensor/ndarray that can all reshape to [N, 3].
        :param fps: Sequence fps.
        """
        is_connected = self.conn is not None
        if not is_connected:
            self.connect()
        for i in range(trans[0].reshape(-1, 3).shape[0]):
            t = time.time()
            self.update_all([r[i] for r in poses], [r[i] for r in trans])
            time.sleep(max(t + 1 / fps - time.time(), 0))
        if not is_connected:
            self.disconnect()
