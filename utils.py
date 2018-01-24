from scipy.linalg import expm, norm
import collections
import itertools
import numpy as np
from torch.autograd import Variable
import torch

def ntuple(n):
    """ Ensure that input has the correct number of elements """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(itertools.repeat(x, n))
    return parse

def getGrid(siz):
    """ Returns grid with coordinates from -siz[0]/2 : siz[0]/2, -siz[1]/2 : siz[1]/2, ...."""
    space = [np.linspace( -(N/2), (N/2), N ) for N in siz]
    mesh = np.meshgrid( *space, indexing='ij' )
    mesh = [np.expand_dims( ax.ravel(), 0) for ax in mesh]

    return np.concatenate(mesh)

def rotate_grid_2D(grid, theta):
    """ Rotate grid """
    theta = np.deg2rad(theta)

    x0 = grid[0, :] * np.cos(theta) - grid[1, :] * np.sin(theta)
    x1 = grid[0, :] * np.sin(theta) + grid[1, :] * np.cos(theta)

    grid[0, :] = x0
    grid[1, :] = x1
    return grid

def rotate_grid_3D(theta, axis, grid):
    """ Rotate grid """
    theta = np.deg2rad(theta)
    axis = np.array(axis)
    rot_mat = expm(np.cross(np.eye(3), axis / norm(axis) * theta))
    rot_mat  =np.expand_dims(rot_mat,2)
    grid = np.transpose( np.expand_dims(grid,2), [0,2,1])

    return np.einsum('ijk,jik->ik',rot_mat,grid)


def get_filter_rotation_transforms(kernel_dims, angles):
    """ Return the interpolation variables needed to transform a filter by a given number of degrees """

    dim = len(kernel_dims)

    # Make grid (centered around filter-center)
    grid = getGrid(kernel_dims)

    # Rotate grid
    if dim == 2:
        grid = rotate_grid_2D(grid, angles)
    elif dim == 3:
        grid = rotate_grid_3D(angles[0], [1, 0, 0], grid)
        grid = rotate_grid_3D(angles[1], [0, 0, 1], grid)


    # Radius of filter
    radius = np.min((np.array(kernel_dims)-1) / 2.)

    #Mask out samples outside circle
    radius = np.expand_dims(radius,-1)
    dist_to_center = np.sqrt(np.sum(grid**2,axis=0))
    mask = dist_to_center>=radius*10

    # Move grid to center
    grid += radius

    return compute_interpolation_grids(grid, kernel_dims, mask=mask)

def compute_interpolation_grids(grid, kernel_dims, mask=None):

    #######################################################
    # The following part is part of nd-linear interpolation

    # Make list where each element represents a dimension
    grid = [grid[i, :] for i in range(grid.shape[0])]

    # Get left and right index (integers)
    inds_0 = [ind.astype(np.integer) for ind in grid]
    inds_1 = [ind + 1 for ind in inds_0]

    # Get weights
    weights = [float_ind - int_ind for float_ind, int_ind in zip(grid, inds_0)]

    # Get samples that are out of bounds
    inds_out_of_bounds = np.logical_or.reduce([ind < 0 for ind in itertools.chain(inds_0, inds_1)] + \
                                              [ind >= siz for ind, siz in zip(inds_0, kernel_dims)] + \
                                              [ind >= siz for ind, siz in zip(inds_1, kernel_dims)])

    # Set these samples to zero get data from upper-left-corner
    for i in range(len(inds_0)):
        inds_0[i][inds_out_of_bounds] = 0
        inds_1[i][inds_out_of_bounds] = 0

    #######################################################
    if mask is not None:
        zero_inds = np.logical_or(inds_out_of_bounds, mask)
    else:
        zero_inds = inds_out_of_bounds
    zero_inds = np.reshape(zero_inds, kernel_dims)


    #Make pytorch-tensors of the interpolation variables
    inds_0 = [torch.LongTensor(ind) for ind in inds_0]
    inds_1 = [torch.LongTensor(ind) for ind in inds_1]
    weights = [Variable(torch.FloatTensor(weight)) for weight in weights]
    zero_inds = [Variable(torch.FloatTensor(zero_inds.astype('float32')))]

    # Uncomment for nearest interpolation (for debugging)
    #inds_1 = [ind*0 for ind in inds_1]
    #weights  = [weight*0 for weight in weights]

    return inds_0, inds_1, weights, zero_inds

def apply_transform(filter, interpolation_variables, filters_size):
    """ Apply a transform specified by the interpolation_variables to a filter """

    dim = 2 if len(filter.size())==4 else 3


    if dim == 2:

        [x0_0, x1_0], [x0_1, x1_1], [w0, w1], zero_inds = interpolation_variables
        zero_inds = zero_inds[0] #Unpack

        rotated_filter = (filter[:, :, x0_0, x1_0] * (1 - w0) * (1 - w1) +
                          filter[:, :, x0_1, x1_0] * w0 * (1 - w1) +
                          filter[:, :, x0_0, x1_1] * (1 - w0) * w1 +
                          filter[:, :, x0_1, x1_1] * w0 * w1)

        rotated_filter = rotated_filter.view(filter.size()[0],filter.size()[1],filters_size[0],filters_size[1])



    elif dim == 3:
        [x0_0, x1_0, x2_0], [x0_1, x1_1, x2_1], [w0, w1, w2], zero_inds = interpolation_variables
        zero_inds = zero_inds[0]  # Unpack

        rotated_filter = (filter[x0_0, x1_0, x2_0] * (1 - w0) * (1 - w1)* (1 - w2) +
                          filter[x0_1, x1_0, x2_0] * w0       * (1 - w1)* (1 - w2) +
                          filter[x0_0, x1_1, x2_0] * (1 - w0) * w1      * (1 - w2) +
                          filter[x0_1, x1_1, x2_0] * w0       * w1      * (1 - w2) +
                          filter[x0_0, x1_0, x2_1] * (1 - w0) * (1 - w1)* w2 +
                          filter[x0_1, x1_0, x2_1] * w0       * (1 - w1)* w2 +
                          filter[x0_0, x1_1, x2_1] * (1 - w0) * w1      * w2 +
                          filter[x0_1, x1_1, x2_1] * w0       * w1      * w2)

        rotated_filter = rotated_filter.view(filter.size()[0], filter.size()[1], filters_size[0], filters_size[1], filters_size[2])

    #Apply mask to zero-pixels that are outside valid region
    rotated_filter = rotated_filter * (1 - zero_inds)
    return rotated_filter



if __name__ == '__main__':
    """ Test rotation of filter """
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.nn.parameter import Parameter
    import math
    from utils import *

    ks = [9,9] #Kernel size
    angle = 45
    interp_vars = get_filter_rotation_transforms(ks, angle)

    w = Variable(torch.zeros([1,1]+ks))
    w[:,:,4,:] = 5
    w[:, :, :, 4] = 1
    w[:,:,0,0] = -1



    print w, apply_transform(w, interp_vars, ks)


