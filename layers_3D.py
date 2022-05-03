#!/usr/bin/env python
__author__ = "Anders U. Waldeland"
__email__ = "anders@nr.no"


"""
This code is an 3D extension of the 2D method proposed in:
Rotation equivariant vector field networks (ICCV 2017)
Diego Marcos, Michele Volpi, Nikos Komodakis, Devis Tuia
https://arxiv.org/abs/1612.09346
https://github.com/dmarcosg/RotEqNet

We use the spherical coordinate system (see https://en.wikipedia.org/wiki/Spherical_coordinate_system)
with coordinates (r/radius, theta/inclination, rho/azimuth). The 3D vector field has the cartesian coordinates (x,y,z)
but we denote them with (u,v,w) in correspondence with the original paper.
"""

import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils import  *





class RotConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, n_inclination = 8, n_azimuth = 4, mode=1):
        super(RotConv, self).__init__()

        kernel_size = ntuple(3)(kernel_size)
        stride = ntuple(3)(stride)
        padding = ntuple(3)(padding)
        dilation = ntuple(3)(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.mode = mode

        #If input is vector field we have two filters (one for each component)
        self.weight1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if self.mode == 2:
            self.weight2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight3 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))


        #Angles (dip and azimuth)
        self.thetas = np.linspace(0, 180, n_inclination, endpoint=False)
        self.phis = np.linspace(0, 360, n_azimuth, endpoint=False)
        self.theta_tensors = []
        self.phi_tensors = []

        #Get interpolation variables
        self.interp_vars = []
        for theta in self.thetas:
            for phi in self.phis:

                self.interp_vars.append(get_filter_rotation_transforms(self.kernel_size, [theta, phi]))#TODO

                self.theta_tensors.append(Variable(torch.FloatTensor([theta / 180. * np.pi])))

                self.phi_tensors.append(Variable(torch.FloatTensor([phi / 180. * np.pi])))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.mode == 2:
            self.weight2.data.uniform_(-stdv, stdv)
            self.weight3.data.uniform_(-stdv, stdv)

    def _apply(self, l):
        # We need to replace tensors and variables with cuda-versions
        # This is most likely not the nicest way to do this but it works...
        self.interp_vars = [[ [l(el2) for el2 in el1 ] for el1 in el0] for el0 in self.interp_vars]
        self.thetas = [l(el) for el in self.thetas]
        self.phis = [l(el) for el in self.phis]

        super(RotConv, self)._apply(l)


    def forward(self,input):

        if self.mode == 1:
            outputs = []

            #Loop through the different filter-transformations
            for ind, interp_vars in enumerate(self.interp_vars):
                #Apply rotation
                weight = apply_transform(self.weight1, interp_vars, self.kernel_size)

                #Do convolution
                out = F.conv3d(input, weight, None, self.stride, self.padding, self.dilation)
                outputs.append(out.unsqueeze(-1))

            #Get the maximum direction (Orientation Pooling)
            strength, max_ind =  torch.max(torch.cat(outputs,-1),-1)

            #Convert to spherical coordinates
            theta = max_ind.float() * (360. / 8. / 180. * np.pi)
            phi = max_ind.float() * (360. / 8. / 180. * np.pi)

            u = F.relu(strength) * torch.sin(theta) * torch.cos(phi)
            v = F.relu(strength) * torch.sin(theta) * torch.sin(phi)
            w = F.relu(strength) * torch.cos(theta)


        if self.mode == 2:
            u = input[0]
            v = input[1]
            w = input[2]

            output_u = []
            output_v = []
            output_w = []
            output_p = [] #magnitude of field

            # Loop through the different filter-transformations
            for ind, interp_vars in enumerate(self.interp_vars):
                theta = self.theta_tensors[ind]
                phi = self.phi_tensors[ind]

                # Apply rotation
                wu = apply_transform(self.weight1, interp_vars, self.kernel_size)
                wv = apply_transform(self.weight2, interp_vars, self.kernel_size)
                ww = apply_transform(self.weight3, interp_vars, self.kernel_size)


                # Do convolution for u
                wru = None#TODO: decompose filters
                u_out = F.conv3d(u, wru, None, self.stride, self.padding, self.dilation)
                output_u.append(u_out.unsqueeze(-1) )

                # Do convolution for v
                wrv = None  # TODO: decompose filters
                v_out = F.conv3d(v, wrv, None, self.stride, self.padding, self.dilation)
                output_v.append(v_out.unsqueeze(-1) )

                # Do convolution for w
                wrw = None  # TODO: decompose filters
                w_out = F.conv3d(w, wrw, None, self.stride, self.padding, self.dilation)
                output_w.append(w_out.unsqueeze(-1))

                #Compute magnitude (p)
                output_p.append( torch.sqrt( v_out**2 + u_out**2 + w_out**2).unsqueeze(-1) )



            # Get the maximum direction (Orientation Pooling)
            strength, max_ind = torch.max(torch.cat(output_p, -1), -1)

            # Select the u,v for the maximum orientation
            u = torch.cat(output_u, -1)
            v = torch.cat(output_v, -1)
            w = torch.cat(output_w, -1)

            u = torch.gather(u, -1, max_ind.unsqueeze(-1))[:, :, :, :, :, 0]
            v = torch.gather(v, -1, max_ind.unsqueeze(-1))[:, :, :, :, :, 0]
            w = torch.gather(w, -1, max_ind.unsqueeze(-1))[:, :, :, :, :, 0]
        
        return u, v, w

class VectorMaxPool(nn.Module):
    def __init__(self,  kernel_size, stride=None, padding=0, dilation=1,
                  ceil_mode=False):
            super(VectorMaxPool, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride or kernel_size
            self.padding = padding
            self.dilation = dilation
            self.ceil_mode = ceil_mode

    def forward(self,input):
        #Assuming input is vector field
        u = input[0]
        v = input[1]
        w = input[2]

        #Magnitude
        p = torch.sqrt( v**2 + u**2 + w**2)
        #Max pool
        _, max_inds = F.max_pool3d(p, self.kernel_size, self.stride,
                     self.padding, self.dilation, self.ceil_mode,
                     return_indices=True)
        #Reshape to please pytorch
        s1 = u.size()
        s2 = max_inds.size()
        
        max_inds = max_inds.view(s1[0], s1[1], s2[2] * s2[3] * s2[4])
        
        u = u.view(s1[0], s1[1], s1[2] * s1[3] * s1[4])
        v = v.view(s1[0], s1[1], s1[2] * s1[3] * s1[4])
        w = w.view(s1[0], s1[1], s1[2] * s1[3] * s1[4])
        
        #Select u/v components according to max pool on magnitude
        u = torch.gather(u, 2, max_inds)
        v = torch.gather(v, 2, max_inds)
        w = torch.gather(w, 2, max_inds)

        #Reshape back
        u = u.view(s1[0], s1[1], s2[2], s2[3], s1[4])
        v = v.view(s1[0], s1[1], s2[2], s2[3], s1[4])
        w = w.view(s1[0], s1[1], s2[2], s2[3], s1[4])

        
        return u,v,w

class Vector2Magnitude(nn.Module):
    def __init__(self):
        super(Vector2Magnitude, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]
        w = input[2]

        p = torch.sqrt(v ** 2 + u ** 2 + w ** 2)
        return p

class VectorBatchNorm(nn.Module):
    def __init__(self):
        super(VectorBatchNorm, self).__init__()

    def forward(self, input):
        if input[0].size()[0] > 1:
            u = input[0]
            v = input[1]
            w = input[2]


            p = torch.sqrt(v ** 2 + u ** 2 + w ** 2)

            #Mean
            mu = torch.mean(p,  0, keepdim=True)
            mu = torch.mean(mu, 2, keepdim=True)
            mu = torch.mean(mu, 3, keepdim=True)
            mu = torch.mean(mu, 4, keepdim=True)

            #Variance
            var = (mu-p)**2
            var = torch.sum(var, 0, keepdim=True)
            var = torch.sum(var, 2, keepdim=True)
            var = torch.sum(var, 3, keepdim=True)
            var = torch.sum(var, 4, keepdim=True)
            std = torch.sqrt(var)

            eps = 0.00001
            std = std + eps


            return u/std, v/std , w/std
        else:
            return input


class VectorUpsampling(nn.Module):
    def __init__(self,  size=None, scale_factor=None, mode = 'trilinear'):
        super(VectorUpsampling, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        # Assuming input is vector field
        u = input[0]
        v = input[1]
        w = input[2]

        u = F.upsample(u, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        v = F.upsample(v, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        w = F.upsample(w, size=self.size, scale_factor=self.scale_factor, mode=self.mode)


        return u, v, w
