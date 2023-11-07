import torch
import torch.nn.functional as F

import diff_operators
import modules, utils

import math
import numpy as np


def initialize_hji_MultiVehicleCollisionNE(dataset, minWith):
    # Initialize the loss function for the multi-vehicle collision avoidance problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    numEvaders = dataset.numEvaders
    num_pos_states = dataset.num_pos_states
    alpha_angle = dataset.alpha_angle
    alpha_time = dataset.alpha_time

    def hji_MultiVehicleCollision(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            # Scale the costate for theta appropriately to align with the range of [-pi, pi]
            dudx[..., num_pos_states:] = dudx[..., num_pos_states:] / alpha_angle

            # Compute the hamiltonian for the ego vehicle
            ham = velocity*(torch.cos(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 0] + torch.sin(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 1]) - omega_max * torch.abs(dudx[..., num_pos_states])

            # Hamiltonian effect due to other vehicles
            for i in range(numEvaders):
                theta_index = num_pos_states+1+i+1
                xcostate_index = 2*(i+1)
                ycostate_index = 2*(i+1) + 1
                thetacostate_index = num_pos_states+1+i
                ham_local = velocity*(torch.cos(alpha_angle*x[..., theta_index]) * dudx[..., xcostate_index] + torch.sin(alpha_angle*x[..., theta_index]) * dudx[..., ycostate_index]) + omega_max * torch.abs(dudx[..., thetacostate_index])
                ham = ham + ham_local

            # Effect of time factor
            ham = ham * alpha_time

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_MultiVehicleCollision


def initialize_hji_air3D(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    alpha = dataset.alpha

    def hji_air3D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        x_u = x * 1.0
        x_u[..., 1] = x_u[..., 1] * alpha['x'] 
        x_u[..., 2] = x_u[..., 2] * alpha['y'] 
        x_u[..., 3] = x_u[..., 3] * alpha['th'] 

        # x_theta = x[..., 3] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        # Scale the coordinates
        # x_theta = alpha['th'] * x_theta

        # Air3D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a

        # Compute the hamiltonian for the ego vehicle
        ham = omega_max * torch.abs(dudx[..., 0] * x_u[..., 2] - dudx[..., 1] * x_u[..., 1] - dudx[..., 2])  # Control component
        ham = ham - omega_max * torch.abs(dudx[..., 2])  # Disturbance component
        ham = ham + (velocity * (torch.cos(x_u[..., 3]) - 1.0) * dudx[..., 0]) + (velocity * torch.sin(x_u[..., 3]) * dudx[..., 1])  # Constant component

        # Effect of time factor
        ham = ham * alpha['time']
        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_air3D

def initialize_hji_air5D(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    omega_max = dataset.omega_max
    acc_max = dataset.acc_max
    alpha = dataset.alpha

    def hji_air5D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        x_u = x * 1.0
        x_u[..., 1] = x_u[..., 1] * alpha['x'] 
        x_u[..., 2] = x_u[..., 2] * alpha['y'] 
        x_u[..., 3] = x_u[..., 3] * alpha['th'] 
        x_u[..., 4] = x_u[..., 4] * alpha['v'] + alpha['v'] 
        x_u[..., 5] = x_u[..., 5] * alpha['v'] + alpha['v']

        # x_theta = x[..., 3] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']
        dudx[..., 4] = dudx[..., 4] / alpha['v']
        # Scale the coordinates
        # x_theta = alpha['th'] * x_theta

        # Air5D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a
        # \dot \v_a = a_a
        # \dot \v_b = a_b 

        # Compute the hamiltonian for the ego vehicle
        ham = omega_max * torch.abs(dudx[..., 0] * x_u[..., 2] - dudx[..., 1] * x_u[..., 1] - dudx[..., 2])  + acc_max * torch.abs(dudx[..., 3])# Control component
        ham = ham - omega_max * torch.abs(dudx[..., 2]) - acc_max * torch.abs(dudx[...,4]) # Disturbance component
        ham = ham + dudx[..., 0]*(-x_u[..., 4] + x_u[..., 5]*torch.cos(x_u[...,3])) + dudx[..., 1]*(x_u[..., 5] * torch.sin(x_u[...,3]))
        # Effect of time factor
        ham = ham * alpha['time']
        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_air5D

