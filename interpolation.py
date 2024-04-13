import torch
import matplotlib.pyplot as plt
from torchrbf import RBFInterpolator
from PIL import Image
import numpy as np
import random

image = Image.open("iphone-apple/rgb/2x/0_00002.png")


xx, yy = np.meshgrid(np.arange(image.width),
                     np.arange(image.height,))



y = torch.tensor(np.stack([xx, yy], axis=-1)).float()

xx = y.reshape(-1,2)[:,0]
yy = y.reshape(-1,2)[:,1]
xx = xx/xx.max()
yy = yy/yy.max()

y = torch.tensor(np.stack([xx, yy], axis=-1)).float().reshape(-1,2)
d = torch.tensor(np.array(image)[:,:,:3]).float().reshape(-1,3)/255.

random_idx = random.sample(list(np.arange(d.shape[0])), 4096)

d = d[random_idx]
y = y[random_idx]

#y = torch.rand(100, 2) # Data coordinates
#d = torch.rand(100, 3) # Data vectors at each point

interpolator = RBFInterpolator(y, d, smoothing=1.0, kernel='thin_plate_spline')

# Query coordinates (100x100 grid of points)
x_ = torch.linspace(0, 1, image.width)
y_ = torch.linspace(0, 1, image.height)
grid_points = torch.meshgrid(x_, y_, indexing='ij')
grid_points = torch.stack(grid_points, dim=-1).reshape(-1, 2)

# Query RBF on grid points
interp_vals = interpolator(grid_points)

# Plot the interpolated values in 2D

plt.imshow(interp_vals.reshape(image.width, image.height, 3))
plt.scatter(y[:, 0]*image.height, y[:, 1]*image.width, c='red', s = 2)
plt.title('Interpolated values in 2D')
plt.savefig('interpolation.png')
