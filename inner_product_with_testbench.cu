
// Import required header files
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel function for inner-product-based matrix multiplication
__global__ void innerProductKernel(float *A, float *B, float *C, int N) {
    // Calculate row and column indices for the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Declare a variable to accumulate the sum
    float sum = 0;

    // Check if the thread's indices are within the matrix dimensions
    if(row < N && col < N) {
        // Compute the dot product for the i-th row of A and the j-th column of B
        for(int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        // Store the sum in the corresponding element of matrix C
        C[row * N + col] = sum;
    }
}

// Function to test the inner product implementation
void testInnerProduct(float *A, float *B, float *C, int N) {
    // Declare device pointers for matrices A, B, and C
    float *d_A, *d_B, *d_C;

    // Error handling for CUDA operations
    cudaError_t err;

    // Allocate memory on the GPU for matrices A, B, and C
    err = cudaMalloc((void **)&d_A, N * N * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaMalloc((void **)&d_B, N * N * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaMalloc((void **)&d_C, N * N * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc C: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the data for matrices A and B from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    dim3 threadsPerBlock(2, 2);
    dim3 blocksPerGrid(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Launch the kernel
    innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the resultant matrix C from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Main function
int main() {
    // Define the dimension of the matrices
    int N = 4;

    // Initialize matrices A and B
    float A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float B[N * N] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    // Declare the result matrix C
    float C[N * N];

    // Run the test function
    testInnerProduct(A, B, C, N);

    // Print the resultant matrix C
    printf("Matrix C: \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Exit the program
    return 0;
}
