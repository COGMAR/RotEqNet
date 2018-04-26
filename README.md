# Rotational Equivariant Vector Field Networks (RotEqNet) for PyTorch

This is an PyTorch implementation of the method proposed in:
Rotation equivariant vector field networks, ICCV 2017,
Diego Marcos, Michele Volpi, Nikos Komodakis, Devis Tuia
https://arxiv.org/abs/1612.09346


The original MATLAB implementation is available at:
https://github.com/dmarcosg/RotEqNet

The main goal is to provide an implementation of the new network layers proposed in the paper. In addition we try to reproduce the results the MNIST-rot dataset to verify the implementation.




### Example usage
```python
from layers_2D import RotConv, VectorMaxPool, VectorBatchNorm, Vector2Magnitude, VectorUpsampling
from torch import nn

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.main = nn.Sequential(           
            RotConv(1, 6, [9, 9], 1, 9 / 2, n_angles=17, mode=1), #First RotConv must have mode=1 
            VectorMaxPool(2),
            VectorBatchNorm(6),
            
            RotConv(6, 16, [9, 9], 1, 9 / 2, n_angles=17, mode=2), #The next RotConv has mode=2 (since input is vector field)
            VectorMaxPool(2),
            VectorBatchNorm(16),
            
            RotConv(16, 32, [9, 9], 1, 1, n_angles=17, mode=2),
            Vector2Magnitude(), #This call converts the vector field to a conventional multichannel image/feature image
            
            nn.Conv2d(32, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.7),
            nn.Conv2d(128, 10, 1),
            
        )

    def forward(self,x):
        x = self.main(x)
        return  x
```


### Dependencies
The following python packages are required

```
torch
numpy
scipy
```
To download and setup the MNIST-rot dataset, cd into the MNIST-folder and run:
```
python download_mnist.py
python make_mnist_rot.py
```
To run the MNIST-test:
```
python mnist_test.py
```
## Results from the MNIST-rot test
The MNIST-experiment from the paper we aim to reproduce is based on
- training on 10 000 images from the MNIST-rot dataset + applying random rotation as augmentation
- validating on 2000 images from the MNIST-rot dataset
- testing on 10 0000 images from the MNIST-rot dataset + with test-time augmentation as described in the paper
- We obtain an test accuracy of 1.2% and the original paper reports 1.1% 

### Known issues:
- The "old" implementation of bilinear interpolation of filters (apply_transformation in utils.py) sometimes cause CUDA runtime error 59. This error disappears when using a torch.gather to collect the samples, but this does reduce the best test error rate to ~3%.  


### Contact
Anders U. Waldeland <br/>
Norwegian Computing Center <br/>
anders@nr.no <br/>



