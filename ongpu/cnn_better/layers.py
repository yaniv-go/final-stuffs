import cupy as cp
import cupyx
import copy
import sys

def get_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  del N
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

  i0 = cp.repeat(cp.arange(field_height), field_width)
  i0 = cp.tile(i0, C)
  i1 = stride * cp.repeat(cp.arange(out_height), out_width)
  j0 = cp.tile(cp.arange(field_width), field_height * C)
  j1 = stride * cp.tile(cp.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = cp.repeat(cp.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_indices(x.shape, field_height, field_width, padding,
                                stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C  , -1)
    return cols

def col2im(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_indices(x_shape, field_height, field_width, padding,
                                stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class GlobalAveragePool:
    def __init__(self, curr_output):
        assert isinstance(curr_output, tuple), 'incorrect input for average pool'

    def test(self, x):
        n, c, w, h = x.shape
        x = x.reshape((n, c, w * h))
        o = cp.mean(x, axis=2, dtype='float32').reshape((n, c, 1, 1))

        return o
    
    def forward(self, x):
        self.mem = n, c, w, h = x.shape
        x = x.reshape(n, c, w * h)
        o = cp.mean(x, axis=2, dtype='float32').reshape((n, c, 1, 1))

        return o
    
    def bprop(self, do):
        n, c, w, h = self.mem

        dx = cp.zeros((n, c, w, h))
        dx[::] = do

        self.mem = [None]

        return dx

    def optimize(self, *args, **kwargs):
        pass