import numpy as np
import os
from mnist import random_rotation, loadMnist



def makeMnistRot():
    """
    Make MNIST-rot from MNIST
    Select all training and test samples from MNIST and select 10000 for train,
    2000 for val and 50000 for test. Apply a random rotation to each image.

    Store in numpy file for fast reading

    """
    np.random.seed(0)

    #Get all samples
    all_samples = loadMnist('train') + loadMnist('test')

    #

    #Empty arrays
    train_data = np.zeros([28,28,10000])
    train_label = np.zeros([10000])
    val_data = np.zeros([28,28,2000])
    val_label = np.zeros([2000])
    test_data = np.zeros([28,28,50000])
    test_label = np.zeros([50000])

    i = 0
    for j in range(10000):
        sample =all_samples[i]
        train_data[:, :, j] =  random_rotation(sample[0])
        train_label[j] = sample[1]
        i += 1

    for j in range(2000):
        sample = all_samples[i]
        val_data[:, :, j] = random_rotation(sample[0])
        val_label[j] = sample[1]
        i += 1

    for j in range(50000):
        sample = all_samples[i]
        test_data[:, :, j] = random_rotation(sample[0])
        test_label[j] = sample[1]
        i += 1


    try:
        os.mkdir('mnist_rot/')
    except:
        None
    np.save('mnist_rot/train_data',train_data)
    np.save('mnist_rot/train_label', train_label)
    np.save('mnist_rot/val_data', val_data)
    np.save('mnist_rot/val_label', val_label)
    np.save('mnist_rot/test_data', test_data)
    np.save('mnist_rot/test_label', test_label)

if __name__ == '__main__':
    makeMnistRot()