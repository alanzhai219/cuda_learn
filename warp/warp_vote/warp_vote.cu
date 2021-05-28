#include <stdio.h>
#include <malloc.h>
#include <cuda_runtime.h>

__global__ void vote_all(int *a, int *b, int n)
{
    int tid = threadIdx.x;
    if (tid > n)
        return;
    int temp = a[tid];
    b[tid] = __all_sync(0xffffffff, temp > 48);// 注意添加了参数 mask
}

__global__ void vote_any(int *a, int *b, int n)
{
    int tid = threadIdx.x;
    if (tid > n)
        return;
    int temp = a[tid];
    b[tid] = __any_sync(0xffffffff, temp > 48);
}

__global__ void vote_ballot(int *a, int *b, int n)
{
    int tid = threadIdx.x;
    if (tid > n)
        return;
    int temp = a[tid];
    b[tid] = __ballot_sync(0xffffffff, temp > 42 && temp < 53);
}

__global__ void vote_union(int *a, int *b, int n)
{
    int tid = threadIdx.x;
    if (tid > n)
        return;
    int temp = a[tid];
    b[tid] = __uni_sync(0xffffffff, temp > 42 && temp < 53);
}

__global__ void vote_active(int *a, int *b, int n)
{
    int tid = threadIdx.x;
    unsigned warpId = tid / warpSize;
    if (tid > n || tid % 2)// 毙掉了所有偶数号线程
        return;
    int temp = a[tid];
    b[warpId] = __activemask() + warpId;
}

int main()
{
    int *h_a, *h_b, *d_a, *d_b;
    int n = 128, m = 32;
    int nsize = n * sizeof(int);

    h_a = (int *)malloc(nsize);
    h_b = (int *)malloc(nsize);
    for (int i = 0; i < n; ++i)
        h_a[i] = i;
    memset(h_b, 0, nsize);
    cudaMalloc(&d_a, nsize);
    cudaMalloc(&d_b, nsize);
    cudaMemcpy(d_a, h_a, nsize, cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, nsize);

    vote_all << <1, n >> >(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
    printf("vote_all():");
    for (int i = 0; i < n; ++i)
    {
        if (!(i % m))
            printf("\n");
        printf("%d ", h_b[i]);
    }
    printf("\n");

    vote_any << <1, n >> >(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
    printf("vote_any():");
    for (int i = 0; i < n; ++i)
    {
        if (!(i % m))
            printf("\n");
        printf("%d ", h_b[i]);
    }
    printf("\n");

    vote_union << <1, n >> >(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
    printf("vote_union():");
    for (int i = 0; i < n; ++i)
    {
        if (!(i % m))
            printf("\n");
        printf("%d ", h_b[i]);
    }
    printf("\n");

    vote_ballot << <1, n >> >(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("vote_ballot():");
    for (int i = 0; i < n; ++i)
    {
        if (!(i % m))
            printf("\n");
        printf("%u ", h_b[i]);// 用无符号整数输出
    }
    printf("\n");

    vote_active << <1, n >> >(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < n/m; i++) {
      printf("vote_active():\n%u ", h_b[i]);// 用无符号整数输出
      printf("\n");
    }

    return 0;
}
