#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

__global__ void reduce_add(int *a, int *b) {
  int tid = threadIdx.x;
  int v = a[tid];
  unsigned mask = __activemask();
  b[tid] = __reduce_add_sync(mask, v);
}

__global__ void reduce_max(int *a, int *b) {
  int tid = threadIdx.x;
  int v = a[tid];
  unsigned mask = __activemask();
  b[tid] = __reduce_max_sync(mask, v);
}

__global__ void reduce_min(int *a, int *b) {
  int tid = threadIdx.x;
  int v = a[tid];
  unsigned mask = __activemask();
  b[tid] = __reduce_min_sync(mask, v);
}

int main(int argc, char *argv[]) {
  // launch params
  int m = 32, n = 128;
  int nsize = n * sizeof(int);

  // host ptr
  int *h_a = (int*)malloc(nsize);
  int *h_b = (int*)malloc(nsize);

  for (int i=0; i<n; i++) {
    h_a[i] = i;
  }
  
  memset(h_b, 0, nsize);

  for (int i=0; i<n; i++) {
    if (!(i%32)) {
      printf("\n");
    }
    printf("%d ", h_a[i]);
  }
  printf("\n");

  // dev ptr
  int *d_a = NULL;
  int *d_b = NULL;
  cudaMalloc((void**)&d_a, nsize);
  cudaMalloc((void**)&d_b, nsize);
  cudaMemcpy(d_a, h_a, nsize, cudaMemcpyHostToDevice);
  cudaMemset(d_b, 0, nsize);

  // launch
  reduce_add<<<1, n>>>(d_a, d_b);

  // copy out
  cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);

  // dump
  printf("reduce and: ");
  for (int i=0; i<n; i++) {
    if (!(i%32)) {
      printf("\n");
    }
    printf("%d ", h_b[i]);
  }
  printf("\n");

  //
  reduce_min<<<1, n>>>(d_a, d_b);
  cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
  
  printf("reduce min: ");
  for (int i=0; i<n; i++) {
    if (!(i%32)) {
      printf("\n");
    }
    printf("%d ", h_b[i]);
  }
  printf("\n");
  
  reduce_max<<<1, n>>>(d_a, d_b);
  cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
  
  printf("reduce max: ");
  for (int i=0; i<n; i++) {
    if (!(i%32)) {
      printf("\n");
    }
    printf("%d ", h_b[i]);
  }
  
  printf("\n");

  free(h_a);
  free(h_b);
  cudaFree(d_a);
  cudaFree(d_b);

  return EXIT_SUCCESS;
}
