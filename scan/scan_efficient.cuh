#include <cuda_runtime.h>

__global__ void prescan(float *g_odata, float *g_idata, int n) {
  extern __shared__ float temp[];// allocated on invocation
  int tid = threadIdx.x;
  // load input into share memory
  temp[2*tid] = g_idata[2*tid];
  temp[2*tid+1] = g_idata[2*tid+1];
  __syncthreads();

  // Up-Sweep
  int offset = 1;
  for (int d = n>>1; d > 0; d >>= 1) {// build sum in place up the tree
    if (tid < d) {
      // calculate index of input arr
      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;

      temp[bi] += temp[ai];
    }
    offset *= 2;
    __syncthreads();
  }

  // Down-Sweep
  // clear the last element
  if (tid == 0) {
    temp[n - 1] = 0;
  } 
  for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  g_odata[2*tid] = temp[2*tid]; // write results to device memory
  g_odata[2*tid+1] = temp[2*tid+1];
}
