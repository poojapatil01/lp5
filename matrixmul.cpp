#include <iostream>

#define N 1024 // Size of the square matrices

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplication(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < N; ++k) {
        sum += a[row * N + k] * b[k * N + col];
    }

    c[row * N + col] = sum;
}

int main() {
    // Allocate memory on host
    int *h_a = new int[N * N];
    int *h_b = new int[N * N];
    int *h_c = new int[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N * N * sizeof(int));
    cudaMalloc((void **)&d_b, N * N * sizeof(int));
    cudaMalloc((void **)&d_c, N * N * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Launch kernel
    matrixMultiplication<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < N * N; ++i) {
        if (h_c[i] != i * i * 4 * N) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Matrix multiplication successful!" << std::endl;
    } else {
        std::cout << "Matrix multiplication failed!" << std::endl;
    }

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
