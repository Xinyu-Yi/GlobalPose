import os
import torch
import numpy as np


__all__ = ['get_kinematic_model', 'get_dynamic_model']


def get_kinematic_model(official_model_file, shape=None):
    import articulate as art
    import carticulate as cart
    if shape is None:
        shape = torch.zeros(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape = shape.to(device)
    model = art.ParametricModel(official_model_file, device=device)
    bone = model.joint_position_to_bone_vector(model.get_zero_pose_joint_and_vertex(shape)[0])[0]

    armature = cart.KinematicArmature()
    armature.n_joints = len(model._J)
    armature.name = os.path.basename(official_model_file).split('.')[0]
    armature.parent = [-1] + model.parent[1:]
    armature.bone = bone.cpu().numpy()

    kinematic_model = cart.KinematicModel(armature)
    return kinematic_model


def get_dynamic_model(official_model_file, shape=None, res=0.02):
    import articulate as art
    import carticulate as cart
    if shape is None:
        shape = torch.zeros(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape = shape.to(device)
    model = art.ParametricModel(official_model_file, device=device)
    mass, com, inertia = model.physical_properties(shape, res=res)
    bone = model.joint_position_to_bone_vector(model.get_zero_pose_joint_and_vertex(shape)[0])[0]

    armature = cart.DynamicArmature()
    armature.n_joints = len(model._J)
    armature.name = os.path.basename(official_model_file).split('.')[0]
    armature.parent = [-1] + model.parent[1:]
    armature.bone = bone.cpu().numpy()
    armature.gravity = np.array((0, -9.8, 0))
    armature.com = com.cpu().numpy()
    armature.mass = mass.cpu().numpy()
    armature.inertia = inertia.cpu().numpy()

    dynamic_model = cart.DynamicModel(armature)
    return dynamic_model
