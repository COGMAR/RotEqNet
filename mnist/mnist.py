import numpy as np
import scipy.misc
import sys
sys.path.append('../')
from utils import getGrid, rotate_grid_2D
from make_mnist_rot import linear_interpolation_2D

def loadMnist(mode):
    print 'Loading MNIST', mode, 'images'
    # Mode = 'train'/'test
    mnist_folder = '/nr/user/andersuw/shared/datasets/mnist/mnist/'

    with file(mnist_folder + mode + '-labels.csv') as f:
        path_and_labels = f.readlines()

    samples = [];
    for entry in path_and_labels:
        path = entry.split(',')[0]
        label = int(entry.split(',')[1])
        img = scipy.misc.imread(mnist_folder + path)

        samples.append([img, label])
    return samples


def linear_interpolation_2D(input_array, indices, outside_val=0, boundary_correction=True):
    # http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy
    output = np.empty(indices[0].shape)
    ind_0 = indices[0,:]
    ind_1 = indices[1,:]

    N0, N1 = input_array.shape

    x0_0 = ind_0.astype(np.integer)
    x1_0 = ind_1.astype(np.integer)
    x0_1 = x0_0 + 1
    x1_1 = x1_0 + 1

    # Check if inds are beyond array boundary:
    if boundary_correction:
        # put all samples outside datacube to 0
        inds_out_of_range = (x0_0 < 0) | (x0_1 < 0) | (x1_0 < 0) | (x1_1 < 0) |  \
                            (x0_0 >= N0) | (x0_1 >= N0) | (x1_0 >= N1) | (x1_1 >= N1)

        x0_0[inds_out_of_range] = 0
        x1_0[inds_out_of_range] = 0
        x0_1[inds_out_of_range] = 0
        x1_1[inds_out_of_range] = 0

    w0 = ind_0 - x0_0
    w1 = ind_1 - x1_0
    # Replace by this...
    # input_array.take(np.array([x0_0, x1_0, x2_0]))
    output = (input_array[x0_0, x1_0] * (1 - w0) * (1 - w1)  +
              input_array[x0_1, x1_0] * w0 * (1 - w1)  +
              input_array[x0_0, x1_1] * (1 - w0) * w1  +
              input_array[x0_1, x1_1] * w0 * w1 )


    if boundary_correction:
        output[inds_out_of_range] = 0

    return output

def loadMnistRot():
    def load_and_make_list(mode):
        data = np.load('mnist_rot/' + mode + '_data.npy')
        lbls = np.load('mnist_rot/' + mode + '_label.npy')
        data = np.split(data, data.shape[2],2)
        lbls = np.split(lbls, lbls.shape[0],0)

        return zip(data,lbls)

    train = load_and_make_list('train')
    val = load_and_make_list('val')
    test = load_and_make_list('test')
    return train, val, test

def random_rotation(data):
    rot = np.random.rand() * 360  # Random rotation
    grid = getGrid([28, 28])
    grid = rotate_grid_2D(grid, rot)
    grid += 13.5
    data = linear_interpolation_2D(data, grid)
    data = np.reshape(data, [28, 28])
    data = data / float(np.max(data))
    return data.astype('float32')
