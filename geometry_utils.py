
from typing import List, Tuple, Union

import math
import quaternion
import pyquaternion
import magnum as mn
import numpy as np

from helpers import pyquaternion_to_quaternion

from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
    quaternion_rotate_vector,
    quaternion_from_coeff,
    quaternion_to_list
)
from habitat_sim.utils import common as utils

def quaternion_xyzw_to_wxyz(v: np.array):
    return np.quaternion(v[3], *v[0:3])

def quaternion_wxyz_to_xyzw(v: np.array):
    return np.quaternion(*v[1:4], v[0])

def quaternion_to_coeff(quat: np.quaternion) -> np.array:
    r"""Converts a quaternions to coeffs in [x, y, z, w] format
    """
    coeffs = np.zeros((4,))
    coeffs[3] = quat.real
    coeffs[0:3] = quat.imag
    return coeffs


def compute_heading_from_quaternion(r):
    """
    r - rotation quaternion
    Computes clockwise rotation about Y.
    """
    # quaternion - np.quaternion unit quaternion
    # Real world rotation
    direction_vector = np.array([0, 0, -1])  # Forward vector
    heading_vector = quaternion_rotate_vector(r.inverse(), direction_vector)

    phi = -np.arctan2(heading_vector[0], -heading_vector[2]).item()
    return phi


def compute_quaternion_from_heading(theta):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    theta - heading angle in radians --- measured clockwise from -Z to X.
    Compute quaternion that represents the corresponding clockwise rotation about Y axis.
    """
    # Real part
    q0 = math.cos(-theta / 2)
    # Imaginary part
    q = (0, math.sin(-theta / 2), 0)

    return np.quaternion(q0, *q)


def compute_egocentric_delta(p1, r1, p2, r2):
    """
    p1, p2 - (x, y, z) position
    r1, r2 - np.quaternions
    Compute egocentric change from (p1, r1) to (p2, r2) in
    the coordinates of (p1, r1)
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    theta_1 = compute_heading_from_quaternion(r1)
    theta_2 = compute_heading_from_quaternion(r2)

    D_rho = math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
    D_phi = (
        math.atan2(x2 - x1, -z2 + z1) - theta_1
    )  # counter-clockwise rotation about Y from -Z to X
    D_theta = theta_2 - theta_1

    return (D_rho, D_phi, D_theta)


def compute_updated_pose(p, r, delta_xz, delta_y):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    p - (x, y, z) position
    r - np.quaternion
    delta_xz - (D_rho, D_phi, D_theta) in egocentric coordinates
    delta_y - scalar change in height
    Compute new position after a motion of delta from (p, r)
    """
    x, y, z = p
    theta = compute_heading_from_quaternion(
        r
    )  # counter-clockwise rotation about Y from -Z to X
    D_rho, D_phi, D_theta = delta_xz

    xp = x + D_rho * math.sin(theta + D_phi)
    yp = y + delta_y
    zp = z - D_rho * math.cos(theta + D_phi)
    pp = np.array([xp, yp, zp])

    thetap = theta + D_theta
    rp = compute_quaternion_from_heading(thetap)

    return pp, rp

def compute_quaternion_from_direction(p1, p2):
    tangent_orientation_matrix = mn.Matrix4.look_at(
        p1, p2, np.array([0, 1.0, 0])
    )

    tangent_orientation_mn = mn.Quaternion.from_matrix(
        tangent_orientation_matrix.rotation()
    )
    tangent_orientation = utils.quat_from_magnum(tangent_orientation_mn)
    
    return tangent_orientation

# Interpolate using rotation vector
def interpolate_quaterions(q1, q2, max_sep=np.pi/6):
    r1 = quaternion.as_rotation_vector(q1)
    r2 = quaternion.as_rotation_vector(q2)

    vertical = np.array([0, 1, 0], dtype=np.float32)
    r1_ = np.dot(r1, vertical) * vertical
    r2_ = np.dot(r2, vertical) * vertical

    rot_angle = np.dot(r2, vertical) - np.dot(r1, vertical)
    complement_angle = (np.pi*2 - np.abs(rot_angle)) * -np.sign(rot_angle)

    rot_angle = rot_angle if np.abs(rot_angle) < np.abs(complement_angle) else complement_angle
    
    angle_thresh = np.pi/6
    rot_inc = np.arange(0, rot_angle, angle_thresh * np.sign(rot_angle))

    if not np.isclose(rot_inc[-1] - rot_angle, 0, atol=1e-3):
        rot_inc = np.concatenate([rot_inc, [rot_angle]])

    rot_val = rot_inc + np.dot(r1_, vertical) 

    rot_quats = list(map(lambda x: quaternion.from_rotation_vector(x * vertical), rot_val))[1:]

    return rot_quats
        
# Interpolate using slerp like quaternion interpolation
def interpolate_quaterions_using_quaternions(q1, q2, max_sep=30):

    q1 = pyquaternion.Quaternion(q1.w, q1.x, q1.y, q1.z)
    q2 = pyquaternion.Quaternion(q2.w, q2.x, q2.y, q2.z)

    q12 = q1.conjugate * q2

    diff = 2 * np.arccos(q12.scalar)
    num_intermediates = (diff * 180 / np.pi) // max_sep

    if num_intermediates == 0:
        return [q2]

    intermediate_py_quat = list(pyquaternion.Quaternion.intermediates(q1, q2, int(num_intermediates), include_endpoints=True))[1:] #Exclude current orientation, but include last

    intermediate_np_quat = list(map(pyquaternion_to_quaternion, intermediate_py_quat)) # Convert to numpy quaternion form

    for i in range(len(intermediate_np_quat)):
        print(quaternion.as_rotation_vector(intermediate_np_quat[i]), end=" ")
    print()

    return intermediate_np_quat
