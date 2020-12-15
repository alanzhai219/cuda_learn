#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif
__global__ void prescan(float *g_odata, float *g_idata, int n) {
  extern __shared__ float temp[];// allocated on invocation
  int thid = threadIdx.x;
  int offset = 1;
  // BLOCK A
  int ai = thid;
  int bi = thid + (n/2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  temp[ai + bankOffsetA] = g_idata[ai];
  temp[bi + bankOffsetB] = g_idata[bi];

  for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (thid < d) {
      // B
      int ai = offset * (2*thid+1) - 1;
      int bi = offset * (2*thid+2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  // C
  if (thid == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n-1)] = 0; }
  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (thid < d) {
      // D
      int ai = offset * (2*thid+1) - 1;
      int bi = offset * (2*thid+2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  // E
  g_odata[ai] = temp[ai + bankOffsetA]; // write results to device memory
  g_odata[bi] = temp[bi + bankOffsetB];
}
