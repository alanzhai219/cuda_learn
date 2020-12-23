#include <stdio.h> 
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 64

int main(int argc, char* argv[]) {
  const int num = N;
  // init and allocate data
  float * arr_in = (float*)malloc(num * sizeof(float));
  float * arr_out = (float*)malloc(num * sizeof(float));
  for (int i=0; i<num; ++i) {
    arr_in[i] = 1.0;
  }
  for (int i=0; i<num; ++i) {
    arr_out[i] = 0.0;
  }

  // malloc in cuda
  float* dev_in = NULL;
  float* dev_out = NULL;
  cudaMalloc((float**)&dev_in, num * sizeof(float));
  cudaMalloc((float**)&dev_out, num * sizeof(float));

  dim3 block(32);
  dim3 grid(2);
  // copy from host to device
  cudaMemcpy(dev_in, arr_in, num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_out, arr_out, num * sizeof(float), cudaMemcpyHostToDevice);

  // call scan func
  prescan<<<grid, block, sizeof(float)*N*2>>>(dev_out, dev_in, num);

  // sync
  cudaDeviceSynchronize();

  cudaMemcpy(arr_out, dev_out, num * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0; i<num; i++) {
    printf("%f\n", arr_out[i]);
  }
  free(arr_in);
  free(arr_out);
  cudaFree(dev_in);
  cudaFree(dev_out);
  return 0;
}
