#include <stdio.h>

__global__ void k(){

  int i = threadIdx.x;
  int j = i;
  if (i<4) j*=2;
  if ((i>3) && (i<8)) j-=(7-i);
  int k = __shfl_sync(0x0FFU, i+100, j);
  printf("lane: %d, result: %d\n", i, k);
}

__forceinline__ __device__ float shuffle(float var, int lane){
   float ret;
   int srcLane = lane;
   int c = 0x1F;
   asm volatile ("shfl.sync.idx.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(srcLane), "r"(c));
  return ret;
}

__global__ void k1(){

  int i = threadIdx.x;
  int j = i;
  if (i<4) j*=2;
  if ((i>3) && (i<8)) j-=(7-i);
  float k = shuffle((float)(i+100), j);
  printf("lane: %d, result: %f\n", i, k);
}

int main(){


  k<<<1,32>>>();
  cudaDeviceSynchronize();
  k1<<<1,8>>>();
  cudaDeviceSynchronize();
}
