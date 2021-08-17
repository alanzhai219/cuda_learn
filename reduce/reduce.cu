#include <iostream>
#include <cuda_runtime.h>

// #include <reduce.cuh>
__global__
void reduce0(int *dI, int *dO) {
  int tid = threadIdx.x;

  int* gd = dI + blockDim.x * blockIdx.x;

  for (int s=1; s<blockIdx.x; s*=2) {
    if ((tid%s) == 0) {
      gd[tid] += gd[tid + s];
    }
  }

  dO[blockDim.x]=gd[0];
}

int main(int argc, char* argv[]) {

  constexpr int N = 2048;
  constexpr int SIZE = N * sizeof(int);

  std::cout << SIZE << std::endl;
  
  // allocate host
  int* hData = (int*)malloc(SIZE);
  // init host
  for (int i=0; i<N; ++i) {
    hData[i] = i+1;
  }
  int* hOut = (int*)malloc(sizeof(int));
  memset((void*)hOut, 0x2, sizeof(int));

  // allocate device
  int* dData = nullptr;
  cudaMalloc((void**)&dData, SIZE);

  int nThreads = 128;
  int nBlocks = (2048 + 128 -1) / nThreads;

  int* dOut = nullptr;
  cudaMalloc((void**)&dOut, nBlocks * sizeof(int));
  cudaMemset(dOut, 0x0, nBlocks * sizeof(int));

  // copy
  cudaMemcpy(dData, hData, SIZE, cudaMemcpyHostToDevice);

  dim3 grid(1,1,nBlocks);
  dim3 block(1,1,nThreads);

  reduce0<<<grid, block>>>(dData, dOut);

  cudaMemcpy(hOut, dOut, nBlocks * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  std::cout << std::hex << *hOut << std::endl;
  // free
  free(hData);
  cudaFree(dData);
  return 0;
}
