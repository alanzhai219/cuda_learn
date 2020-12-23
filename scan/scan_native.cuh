__global__ void scan_native(float *g_odata, float *g_idata, int n) {
  extern __shared__ float temp[];
  
  int tid = threadIdx.x;

  int pout = 0, pin = 1;

  temp[pout * n + tid] = (tid>0) ? g_idata[tid-1] : 0;
  __syncthreads();
  for (int offset = 1; offset < n; offset=*2) {
    pout = 1 - pout;
    pin = 1 - pout;
    int baseout = pout * n;
    int basein = pin * n;
    if (tid >= offset) {
      temp[baseout + tid]  = temp[basein + tid] + temp[basein + tid + offset];
    } else {
      temp[baseout + tid]  = temp[basein + tid];
    }
  __syncthreads();
  }
  g_odata[tid]  = temp[baseout + tid];
}
