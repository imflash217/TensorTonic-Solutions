#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add_kernel(const float *A, const float *B, float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

void practice_raw_cuda(const float *host_A, const float *host_B, float *host_C, int N) {
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    size_t bytes = N * sizeof(float);

    // [Practice] Allocate memory on the GPU device
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // [Practice] Copy data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_A, host_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, bytes, cudaMemcpyHostToDevice);

    // Configure execution parameters
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Launch the kernel using your manually allocated pointers
    vector_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // [Practice] Copy the result back from Device (GPU) to Host (CPU)
    cudaMemcpy(host_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // [Practice] Free the allocated device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// TensorTonic Entry Point
extern "C" void solve(const float *A, const float *B, float *C, int N) {
    // TensorTonic passes pre-allocated device pointers by default. 
    // To practice the raw pipeline, we must simulate having host pointers first.
    
    size_t bytes = N * sizeof(float);
    float *host_A = (float*)malloc(bytes);
    float *host_B = (float*)malloc(bytes);
    float *host_C = (float*)malloc(bytes);

    // Pull the platform's test data back to the CPU temporarily 
    cudaMemcpy(host_A, A, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_B, B, bytes, cudaMemcpyDeviceToHost);

    // Run your raw manual pipeline!
    practice_raw_cuda(host_A, host_B, host_C, N);

    // Send the final result back to TensorTonic's expected output pointer
    cudaMemcpy(C, host_C, bytes, cudaMemcpyHostToDevice);

    // Clean up host memory
    free(host_A);
    free(host_B);
    free(host_C);
}