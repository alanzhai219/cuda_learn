#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel1() //没有bank conflict
{
    int tid=threadIdx.x;
    __shared__ short cache[128];
    cache[tid*1]=1;
    // printf("bank id = %d\n", (tid / 4 )%32);
    short number=cache[tid*1];
}

__global__ void kernel2() //有bank conflict
{
    int tid=threadIdx.x;
    __shared__ int cache[128];
    cache[tid*4]=1;
    printf("bank id = %d\n", (tid*4 * sizeof(int)/ 4 )%32);
    int number=cache[tid*4];
}

int main()
{
  cudaSetDevice(0);
  int ret = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  printf("%d\n", ret);
  printf("%d\n", sizeof(short));

  cudaSharedMemConfig MemConfig;
  cudaDeviceGetSharedMemConfig(&MemConfig);
  printf("--------------------------------------------\n");
  printf("%d\n", MemConfig);
  switch (MemConfig) {
      case cudaSharedMemBankSizeFourByte:
        printf("the device is cudaSharedMemBankSizeFourByte: 4-Byte\n");
      break;
      case cudaSharedMemBankSizeEightByte:
        printf("the device is cudaSharedMemBankSizeEightByte: 8-Byte\n");
      break;
  }

  kernel1<<<1,128>>>();
  kernel2<<<1,32>>>();
  return 0;
}
