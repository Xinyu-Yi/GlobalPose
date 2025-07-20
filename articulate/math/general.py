r"""
    General math utils.
"""


__all__ = ['lerp', 'normalize_tensor', 'append_value', 'append_zero', 'append_one', 'vector_cross_matrix',
           'vector_cross_matrix_np', 'block_diagonal_matrix_np', 'hat', 'vee']


import numpy as np
import torch
from functools import partial


def lerp(a, b, t):
    r"""
    Linear interpolation (unclamped).

    :param a: Begin value.
    :param b: End value.
    :param t: Lerp weight. t = 0 will return a; t = 1 will return b.
    :return: The linear interpolation value.
    """
    return a * (1 - t) + b * t


def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False, avoid_nan=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :param avoid_nan: If True, return zeros if norm is 0.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    if avoid_nan:
        normalized_x[torch.isnan(normalized_x)] = 0
    return normalized_x if not return_norm else (normalized_x, norm)


def append_value(x: torch.Tensor, value: float, dim=-1):
    r"""
    Append a value to a tensor in a specific dimension. (torch)

    e.g. append_value(torch.zeros(3, 3, 3), 1, dim=1) will result in a tensor of shape [3, 4, 3] where the extra
         part of the original tensor are all 1.

    :param x: Tensor in any shape.
    :param value: The value to be appended to the tensor.
    :param dim: The dimension to be expanded.
    :return: Tensor in the same shape except for the expanded dimension which is 1 larger.
    """
    app = torch.ones_like(x.index_select(dim, torch.tensor([0], device=x.device))) * value
    x = torch.cat((x, app), dim=dim)
    return x


append_zero = partial(append_value, value=0)
append_one = partial(append_value, value=1)


def vector_cross_matrix(x: torch.Tensor):
    r"""vector_cross_matrix() is deprecated, use hat() instead"""
    return hat(x.view(-1, 3))


def vector_cross_matrix_np(x):
    r"""
    Get the skew-symmetric matrix :math:`[v]_\times\in so(3)` for vector3 `v`. (numpy, single)

    :param x: Vector3 in shape [3].
    :return: The skew-symmetric matrix in shape [3, 3].
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]], dtype=float)


def block_diagonal_matrix_np(matrix2d_list):
    r"""
    Generate a block diagonal 2d matrix using a series of 2d matrices. (numpy, single)

    :param matrix2d_list: A list of matrices (2darray).
    :return: The block diagonal matrix.
    """
    ret = np.zeros(sum([np.array(m.shape) for m in matrix2d_list]))
    r, c = 0, 0
    for m in matrix2d_list:
        lr, lc = m.shape
        ret[r:r+lr, c:c+lc] = m
        r += lr
        c += lc
    return ret


def vee(m: torch.Tensor):
    r"""
    Return the 3D vector of the 3x3 skew-symmetric matrix. (torch, batch)

    :param m: Tensor in shape [..., 3, 3].
    :return: Tensor in shape [..., 3].
    """
    m = (m - m.transpose(-1, -2)) / 2
    return torch.stack((m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]), dim=-1)


def hat(v: torch.Tensor):
    r"""
    Return the 3x3 skew-symmetric matrix of the 3D vector. (torch, batch)

    :param v: Tensor in shape [..., 3].
    :return: Tensor in shape [..., 3, 3].
    """
    return torch.stack((torch.zeros_like(v[..., 0]), -v[..., 2], v[..., 1],
                        v[..., 2], torch.zeros_like(v[..., 0]), -v[..., 0],
                        -v[..., 1], v[..., 0], torch.zeros_like(v[..., 0]),), dim=-1).view(*v.shape[:-1], 3, 3)
