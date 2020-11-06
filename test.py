import numpy as np

"""
a = np.arange(9) + 1
a = np.array((a, np.roll(a, 1), np.roll(a, 2))).T
k = 3
p = 0 ; c = 3
x = np.zeros(a.shape).T

for j in range(a.shape[1]):
    for i in range(a.shape[0]):    
        p = i // k 

        m = i - 3 * p
        n = j * c + p
        x[m, n] = a[i, j]
        print (x)

        if i == 8: p = 0
print (a)
print (x)
"""
"""
best memory wise
data = np.arange(16 * 3)
b = np.ones(data.shape).astype(bool)
i = 0 ; x = data[0] ; k = 4 ; f = 3
while np.any(b):
    if b[i] == 0: i = np.argmax(b > 0) ; x = data[i] ; continue
    b[i] = 0
    j = i % k * k * f + i // k
    data[j], x = x, data[j]     
    i = j
print (data.reshape(4, 12))
""" 
"""
best speed
data = np.arange(27)
k = 3
x = np.arange(0, data.shape[0], k)
x = np.concatenate((x, x + 1, x + 2))
data = data[x]
del x
print (data.reshape(3, 9))
"""
def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1
    
    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

def col2im_sliding(B, block_size, image_size):
    m,n = block_size
    mm,nn = image_size
    return B.reshape(nn-n+1,mm-m+1).T 
      # Or simply B.reshape(nn-n+1,-1).T
      # Or B.reshape(mm-m+1,nn-n+1,order='F'

def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d. 
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
  
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

a = Fc(1, 1, 1)
print (a.forward(np.array([[1]])))



    
