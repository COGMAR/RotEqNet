from __future__ import division
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
    mask = dist_to_center>=radius+.0001
    mask = 1-mask

    # Move grid to center
    grid += radius

    return compute_interpolation_grids(grid, kernel_dims, mask)

def compute_interpolation_grids(grid, kernel_dims, mask):

    #######################################################
    # The following part is part of nd-linear interpolation

    #Add a small eps to grid so that floor and ceil operations become more stable
    grid += 0.000000001

    # Make list where each element represents a dimension
    grid = [grid[i, :] for i in range(grid.shape[0])]

    # Get left and right index (integers)
    inds_0 = [ind.astype(np.integer) for ind in grid]
    inds_1 = [ind + 1 for ind in inds_0]

    # Get weights
    weights = [float_ind - int_ind for float_ind, int_ind in zip(grid, inds_0)]

    # Special case for when ind_1 == size (while ind_0 == siz)
    # In that case we select ind_0
    ind_1_out_of_bounds = np.logical_or.reduce([ind == siz for ind, siz in zip(inds_1, kernel_dims)])
    for i in range(len(inds_1)):
        inds_1[i][ind_1_out_of_bounds] = 0


    # Get samples that are out of bounds or outside mask
    inds_out_of_bounds = np.logical_or.reduce([ind < 0 for ind in itertools.chain(inds_0, inds_1)] + \
                                              [ind >= siz for ind, siz in zip(inds_0, kernel_dims)] + \
                                              [ind >= siz for ind, siz in zip(inds_1, kernel_dims)] +
                                              (1-mask).astype('bool')
                                              )


    # Set these samples to zero get data from upper-left-corner (which will be put to zero)
    for i in range(len(inds_0)):
        inds_0[i][inds_out_of_bounds] = 0
        inds_1[i][inds_out_of_bounds] = 0

    #Reshape
    inds_0 = [np.reshape(ind,[1,1]+kernel_dims) for ind in inds_0]
    inds_1 = [np.reshape(ind,[1,1]+kernel_dims) for ind in inds_1]
    weights = [np.reshape(weight,[1,1]+kernel_dims)for weight in weights]

    #Make pytorch-tensors of the interpolation variables
    inds_0 = [Variable(torch.LongTensor(ind)) for ind in inds_0]
    inds_1 = [Variable(torch.LongTensor(ind)) for ind in inds_1]
    weights = [Variable(torch.FloatTensor(weight)) for weight in weights]

    #Make mask pytorch tensor
    mask = mask.reshape(kernel_dims)
    mask = mask.astype('float32')
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, 0)
    mask = torch.FloatTensor(mask)

    # Uncomment for nearest interpolation (for debugging)
    #inds_1 = [ind*0 for ind in inds_1]
    #weights  = [weight*0 for weight in weights]

    return inds_0, inds_1, weights, mask

def apply_transform(filter, interp_vars, filters_size, old_bilinear_interpolation=True):
    """ Apply a transform specified by the interpolation_variables to a filter """

    dim = 2 if len(filter.size())==4 else 3

    if dim == 2:


        if old_bilinear_interpolation:
            [x0_0, x1_0], [x0_1, x1_1], [w0, w1] = interp_vars
            rotated_filter = (filter[:, :, x0_0, x1_0] * (1 - w0) * (1 - w1) +
                          filter[:, :, x0_1, x1_0] * w0 * (1 - w1) +
                          filter[:, :, x0_0, x1_1] * (1 - w0) * w1 +
                          filter[:, :, x0_1, x1_1] * w0 * w1)
        else:

            # Expand dimmentions to fit filter
            interp_vars = [[inner_el.expand_as(filter) for inner_el in outer_el] for outer_el in interp_vars]

            [x0_0, x1_0], [x0_1, x1_1], [w0, w1] = interp_vars

            a = torch.gather(torch.gather(filter, 2, x0_0), 3, x1_0) * (1 - w0) * (1 - w1)
            b = torch.gather(torch.gather(filter, 2, x0_1), 3, x1_0)* w0 * (1 - w1)
            c = torch.gather(torch.gather(filter, 2, x0_0), 3, x1_1)* (1 - w0) * w1
            d = torch.gather(torch.gather(filter, 2, x0_1), 3, x1_1)* w0 * w1
            rotated_filter = a+b+c+d

        rotated_filter = rotated_filter.view(filter.size()[0],filter.size()[1],filters_size[0],filters_size[1])

    elif dim == 3:
        [x0_0, x1_0, x2_0], [x0_1, x1_1, x2_1], [w0, w1, w2] = interp_vars

        rotated_filter = (filter[x0_0, x1_0, x2_0] * (1 - w0) * (1 - w1)* (1 - w2) +
                          filter[x0_1, x1_0, x2_0] * w0       * (1 - w1)* (1 - w2) +
                          filter[x0_0, x1_1, x2_0] * (1 - w0) * w1      * (1 - w2) +
                          filter[x0_1, x1_1, x2_0] * w0       * w1      * (1 - w2) +
                          filter[x0_0, x1_0, x2_1] * (1 - w0) * (1 - w1)* w2 +
                          filter[x0_1, x1_0, x2_1] * w0       * (1 - w1)* w2 +
                          filter[x0_0, x1_1, x2_1] * (1 - w0) * w1      * w2 +
                          filter[x0_1, x1_1, x2_1] * w0       * w1      * w2)

        rotated_filter = rotated_filter.view(filter.size()[0], filter.size()[1], filters_size[0], filters_size[1], filters_size[2])

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

    w = Variable(torch.ones([1,1]+ks))
    #w[:,:,4,:] = 5
    w[:, :, :, 4] = 5
    #w[:,:,0,0] = -1


    print(w)
    for angle in [0,90,45,180,65,10]:
        print(angle,'degrees')
        print(apply_transform(w, get_filter_rotation_transforms(ks, angle)[:-1], ks,old_bilinear_interpolation=True) * Variable(get_filter_rotation_transforms(ks, angle)[-1]))
        print('Difference', torch.sum(apply_transform(w, get_filter_rotation_transforms(ks, angle)[:-1], ks,old_bilinear_interpolation=False) * Variable( get_filter_rotation_transforms(ks, angle)[-1]) - apply_transform(w, get_filter_rotation_transforms(ks, angle)[:-1], ks,old_bilinear_interpolation=True) * Variable(get_filter_rotation_transforms(ks, angle)[-1])))


