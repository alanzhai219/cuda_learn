#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <numeric>

#include <cuda_runtime.h>

__global__ void vote_active() {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int fullmask = __activemask();
  printf("fullmask -- %x\n", fullmask);
  if (tid%2 == 0) {
    int mask = __activemask();
    printf("mask -- %x\n", mask);
  }
}

int main(int argc, char* argv[]) {
    /*
    const int n = 128;
    const int m = 32;

    // define ptr in host and device
    int *h_a = static_cast<int*>(malloc(n*sizeof(int)));
    int *h_b = static_cast<int*>(malloc(m*sizeof(int)));
    int *d_a = nullptr;
    int *d_b = nullptr;

    // initialize data in host
    std::iota(h_a, h_a+n, 0);
    std::cout << "h_a data:";
    for (auto i=0; i<n; ++i) {
        if (i%32 == 0) {
            std::cout << std::endl;
        }
        std::cout << std::setw(3) << h_a[i] << " ";
    }

    cudaMalloc(&d_a, n * sizeof(int));
    cudaMemcpy(d_a, h_a, n*sizeof(int), cudaMemcpyHostToDevice);
    */
    std::cout << "<<<1,64>>>\n";
    dim3 gridSize0(1);
    dim3 blockSize0(64);
    vote_active<<<gridSize0, blockSize0>>>();
    cudaDeviceSynchronize();
    
    std::cout << "<<<1,48>>>\n";
    dim3 gridSize1(1);
    dim3 blockSize1(48);
    vote_active<<<gridSize1, blockSize1>>>();
    cudaDeviceSynchronize();

    std::cout << "branch <<<1,48>>>\n";
    vote_active<<<gridSize1, blockSize1>>>();
    cudaDeviceSynchronize();
    std::cout << "\n=====end=====\n";
    return 0;
}