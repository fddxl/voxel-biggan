import binvox
import numpy as np


def read_binvox(fn):
    with open(fn, 'rb') as f:
        model = binvox.read_as_3d_array(f)
        return model.data.astype(np.float32)


def save_binvox(fn, data):
    dims = data.shape
    translate = [0.0, 0.0, 0.0]
    model = binvox.Voxels(data, dims, translate, 1.0, 'xyz')
    with open(fn, 'wb') as f:
        model.write(f)
