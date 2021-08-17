#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <numeric>

#include <cuda_runtime.h>

__global__ void vote_active() {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int fullmask = __activemask();
  printf("tid = %d, fullmask = %x\n", tid, fullmask);
  if (tid%2 == 0) {
    int mask = __activemask();
    printf("tid = %d, mask = %x\n", tid, mask);
  }
}

int main(int argc, char* argv[]) {

    dim3 gridSize1(1);
    dim3 blockSize1(48);

    std::cout << "branch <<<1,48>>>\n";
    vote_active<<<gridSize1, blockSize1>>>();
    cudaDeviceSynchronize();
    std::cout << "\n=====end=====\n";
    return 0;
}