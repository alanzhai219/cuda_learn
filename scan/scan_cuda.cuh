__global__ void scan_cuda(float *g_out, float *g_in, int n) {
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
