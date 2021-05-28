#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

__global__ void match_all(int* a, int* b) {
  int tid = threadIdx.x;
  int tmp = a[tid];
  int pred;
  b[tid] = __match_all_sync(0xffffffff, tmp, &pred);
  printf("laneId=%d, %d\n", tid/warpSize, pred);
}

__global__ void match_any(int* a, int* b) {
  int tid = threadIdx.x;
  int tmp = a[tid];
  b[tid] = __match_any_sync(0xffffffff, tmp);
}

int main() {
  // input config
  int m = 32, n = 128;
  int nsize = n * sizeof(int);

  // ptr
  int* h_a, *h_b;

  h_a = (int *)malloc(nsize);
  h_b = (int *)malloc(nsize);
  for (int i = 0; i < n/2; i++) {
    h_a[i] = 0;
    h_a[i+n/2] = 1;
  }
  h_a[30] = 2;
  h_a[31] = 3;
  memset(h_b, 0, nsize);

  // gpu ptr
  int* d_a, *d_b;
  cudaMalloc((void**)&d_a, nsize);
  cudaMalloc((void**)&d_b, nsize);
  cudaMemcpy(d_a, h_a, nsize, cudaMemcpyHostToDevice);
  cudaMemset(d_b, 0, nsize);

  // launch kernel
  match_any<<<1, n>>>(d_a, d_b);

  // copy out
  cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);

  // print
  printf("match any:");
  for (int i=0; i<n; i++) {
    if (i%m == 0) {
      printf("\n");
    }
    printf("%u ", h_b[i]);
  }
  printf("\n");

  printf("-------------------------------------------------\n");

  // launch kernel
  cudaMemset(d_b, 0, nsize);
  match_all<<<1, n>>>(d_a, d_b);

  // copy
  cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);

  // print
  printf("\n");
  printf("match all:");
  for (int i=0; i<n; i++) {
    if (i%m == 0) {
      printf("\n");
    }
    printf("%u ", h_b[i]);
  }
  printf("\n");
  return EXIT_SUCCESS;
}
