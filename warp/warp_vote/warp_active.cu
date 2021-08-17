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