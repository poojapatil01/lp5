import numpy as np
from numba import cuda

@cuda.jit
def vector_add(a, b, c):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < len(c):
        c[idx] = a[idx] + b[idx]

# Size of the vectors
n = 1000000

# Generate random vectors
a = np.random.randint(1, 1000, size=n)
b = np.random.randint(1, 1000, size=n)
c = np.zeros_like(a)

# Allocate memory on GPU
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(c)

# Launch kernel
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Copy result back to host
d_c.copy_to_host(c)

# Verify result (print first 10 elements)
print(c[:10])











import numpy as np
from numba import cuda

@cuda.jit
def matrix_mul(a, b, c, N):
    row, col = cuda.grid(2)
    if row < N and col < N:
        sum_val = 0
        for k in range(N):
            sum_val += a[row, k] * b[k, col]
        c[row, col] = sum_val

# Size of the square matrices
N = 512

# Generate random matrices
a = np.random.randint(1, 1000, size=(N, N))
b = np.random.randint(1, 1000, size=(N, N))
c = np.zeros((N, N), dtype=np.int32)

# Allocate memory on GPU
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(c)

# Define block and grid dimensions
threads_per_block = (16, 16)
blocks_per_grid = ((N + threads_per_block[0] - 1) // threads_per_block[0], 
                   (N + threads_per_block[1] - 1) // threads_per_block[1])

# Launch kernel
matrix_mul[blocks_per_grid, threads_per_block](d_a, d_b, d_c, N)

# Copy result back to host
d_c.copy_to_host(c)

# Verify result (print some elements)
print(c[:2, :2])