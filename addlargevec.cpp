#include <iostream>

// CUDA kernel for vector addition
__global__ void vectorAddition(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int size = 1000000; // Size of the vectors

    // Allocate memory on host
    int *h_a = new int[size];
    int *h_b = new int[size];
    int *h_c = new int[size];

    // Initialize vectors
    for (int i = 0; i < size; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size * sizeof(int));
    cudaMalloc((void **)&d_b, size * sizeof(int));
    cudaMalloc((void **)&d_c, size * sizeof(int));

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch kernel
    vectorAddition<<<numBlocks, blockSize>>>(d_a, d_b, d_c, size);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
    } else {
        std::cout << "Vector addition failed!" << std::endl;
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
