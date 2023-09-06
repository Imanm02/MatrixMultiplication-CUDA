# Matrix Multiplication in CUDA

Here's the CUDA matrix multiplication implementation using two approaches: inner product and outer product. Matrix multiplication is a fundamental operation in linear algebra and has various applications in computer science and data analysis.

## Code Walkthrough

### CUDA Matrix Multiplication with Inner Product Approach

This notebook demonstrates matrix multiplication using the inner product approach on a CUDA GPU. The code consists of two major parts:

#### Kernel Function (`innerProductKernel`)

The `innerProductKernel` function is the GPU kernel that carries out the matrix multiplication. Each thread calculates one element of the resultant matrix \( C \) by taking the dot product of one row of matrix \( A \) with one column of matrix \( B \).

- `__global__ void innerProductKernel(float *A, float *B, float *C, int N)`: This is the CUDA kernel function where the actual multiplication occurs.

    - `float *A, *B, *C`: Pointers to matrices \( A \), \( B \), and \( C \) stored in the device memory.
  
    - `int N`: Dimension of the square matrices \( A \), \( B \), and \( C \).
    
    - `row` and `col`: Calculated to determine each thread's row and column index.
    
    - `sum`: Variable to store the intermediate sum while calculating each element \( C_{ij} \).
  
#### Test Function (`testInnerProduct`)

The `testInnerProduct` function serves as a testbench. It allocates device memory, copies the input matrices from host to device, launches the kernel, and then copies the resultant matrix back to the host.

- `void testInnerProduct(float *A, float *B, float *C, int N)`: Function that prepares the data, calls the CUDA kernel and retrieves the data.

    - `float *d_A, *d_B, *d_C`: Device pointers for matrices \( A \), \( B \), and \( C \).
  
    - `cudaMalloc`: Allocates memory on the device.
    
    - `cudaMemcpy`: Transfers data between host and device.
    
    - `innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N)`: Kernel invocation.
    
    - `cudaFree`: Frees the allocated device memory.

#### Compilation and Execution

The code is compiled using the NVIDIA CUDA Compiler (`nvcc`) and executed on the GPU. The resultant matrix \( C \) is then printed on the console.

To run this part of the code:

1. Use the `%%writefile` magic command to write the CUDA code into a `.cu` file.
2. Use `!nvcc` to compile the code.
3. Run the compiled executable with `!./inner_product_with_testbench`.

This approach utilizes the parallel processing capabilities of a GPU to accelerate matrix multiplication.

### CUDA Matrix Multiplication with Outer Product Approach

This notebook demonstrates how to perform matrix multiplication using the outer product approach on a GPU using CUDA. The code consists of two major parts:

#### Kernel Function (`outerProductKernel`)

The `outerProductKernel` function is the GPU kernel that performs the matrix multiplication in an outer-product-based approach. In this method, each thread is responsible for calculating an entire row-column multiplication rather than a single element of the resultant matrix \( C \).

- `__global__ void outerProductKernel(float *A, float *B, float *C, int N)`: This is the CUDA kernel function where the actual multiplication takes place.

    - `float *A, *B, *C`: Pointers to matrices \( A \), \( B \), and \( C \) stored in the device memory.
  
    - `int N`: Dimension of the square matrices \( A \), \( B \), and \( C \).
    
    - `row` and `col`: Calculated to determine the row and column index for each thread.
    
    - `temp_C`: Variable to store the intermediate results of the outer product of two vectors for a specific element in \( C \).

#### Test Function (`testOuterProduct`)

The `testOuterProduct` function serves as a testbench. It allocates device memory, copies the input matrices from host to device, launches the kernel, and then copies the resultant matrix back to the host.

- `void testOuterProduct(float *A, float *B, float *C, int N)`: Function that prepares the data, calls the CUDA kernel and retrieves the data.

    - `float *d_A, *d_B, *d_C`: Device pointers for matrices \( A \), \( B \), and \( C \).
  
    - `cudaMalloc`: Allocates memory on the device.
    
    - `cudaMemcpy`: Transfers data between host and device.
    
    - `outerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N)`: Kernel invocation.
    
    - `cudaFree`: Frees the allocated device memory.

#### Compilation and Execution

The code is compiled using the NVIDIA CUDA Compiler (`nvcc`) and executed on the GPU. The resultant matrix \( C \) is then printed on the console.

To run this part of the code:

1. Use the `%%writefile` magic command to write the CUDA code into a `.cu` file.
2. Use `!nvcc` to compile the code.
3. Run the compiled executable with `!./outer_product_with_testbench`.

This approach leverages the parallel processing power of a GPU to execute the outer-product-based matrix multiplication efficiently.
