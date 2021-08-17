#include <stdio.h>

#include <stdlib.h>

#include <cuda.h>

/* E.D. Riedijk */

__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__global__ void kern(int *sm){

   if (threadIdx.x==0)

      sm[blockIdx.x]=get_smid();

}

int main(int argc, char *argv[]){

   int N = atoi(argv[1]);

   int *sm = NULL;
   int *sm_d = NULL;

   sm = (int *) malloc(N * sizeof(int));

   cudaMalloc((void**)&sm_d, N*sizeof(int));

   kern<<<N, N>>>(sm_d);

   cudaMemcpy(sm, sm_d, N*sizeof(int), cudaMemcpyDeviceToHost);

   for (int i=0;i<N;i++)

      printf("%d %d\n",i,sm[i]);

return 0;

}
