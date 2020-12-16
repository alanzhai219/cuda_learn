#include <stdio.h> 
#include <cuda_runtime.h>

#define N 32
/*
__global__ void scan(float *g_odata, float *g_idata, int n) {
  __shared__ float temp[2*N+1];
  // allocated on invocation   
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // Load input into shared memory.
  // This is exclusive scan, so shift right by one
  // and set first element to 0
  temp[tid] = (tid == 0) ? 0 else g_idata[tid-1];
  __syncthreads();
  for (int offset = 1; offset < n; offset *= 2)   {
    if (tid >= offset) {
      temp[n + tid] += temp[tid - offset];
    } else { 
      temp[n + tid] = temp[tid];
    }
    __syncthreads();
  }   
  g_odata[tid] = temp[n+tid];
  // write output
} 
*/

__global__ void scan_native(float *g_out, float *g_in, int n) {
  __shared__ float tmp[N];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  tmp[tid] = g_in[tid];
  __syncthreads();
  for (int offset=1; offset<n; offset *= 2) {
    if (tid >= offset) {
      tmp[tid] += tmp[tid-offset];
    } else {
      tmp[tid] = tmp[tid];
    }
    __syncthreads();
  }
  g_out[tid] = tmp[tid];
}

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
  dim3 grid(1);
  // copy from host to device
  cudaMemcpy(dev_in, arr_in, num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_out, arr_out, num * sizeof(float), cudaMemcpyHostToDevice);

  // call scan func
  scan_native<<<grid, block>>>(dev_out, dev_in, num);

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
