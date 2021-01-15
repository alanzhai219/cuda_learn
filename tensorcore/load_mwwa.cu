#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

__global__ void load_matrix(half *A, half *B, float *C) {
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c;

  // load
  nvcuda::wmma::load_matrix_sync(frag_a, A, 16);
  nvcuda::wmma::load_matrix_sync(frag_b, B, 16);
  nvcuda::wmma::fill_fragment(frag_c, 0.0f);

  // compute
  nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

  // save
  nvcuda::wmma::store_matrix_sync(C, frag_c, 16, nvcuda::wmma::mem_row_major);
}

void init_A(half *matrix, int N) {
  for (int i=0; i<N; i++) {
    matrix[i] = i + 0.0f;
  }
}

template <class T>
void init_const(T *matrix, float v, int N) {
  // memset(matrix, (T)v, N*sizeof(T));
  for (int i=0; i<N; i++) {
    matrix[i] = v;
  }
}

int main(int argc, char **agrv) {
  int N=16, M=16, K=16;
  int asize = N * K * sizeof(half);
  int bsize = K * M * sizeof(half);

  int csize = N * M * sizeof(float);

  // host ptr
  half *h_a = (half *)malloc(asize);
  half *h_b = (half *)malloc(bsize);
  float *h_c = (float *)malloc(csize);

  // initialize
  // init_A(h_a, N * K);
  init_const(h_a, 1.0f, N * K);
  init_const(h_b, 1.0f, K * M);

  init_const(h_c, 1.0f, N * M);

  // dev ptr
  half *d_a, *d_b;
  float *d_c;
  cudaMalloc((void**)&d_a, asize);
  cudaMalloc((void**)&d_b, bsize);
  cudaMalloc((void**)&d_c, csize);

  // cp
  cudaMemcpy(d_a, h_a, asize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bsize, cudaMemcpyHostToDevice);
  cudaMemset(d_c, 0, csize);

  // launch
  load_matrix<<<1, 32>>>(d_a, d_b, d_c);

  // copy out
  cudaMemcpy(h_c, d_c, csize, cudaMemcpyDeviceToHost);

  // printf
  for (int i=0; i<10; i++) {
    printf("%f \n", h_c[i]);
  }
  printf("\n");

  free(h_a);
  free(h_b);
  free(h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return EXIT_SUCCESS;
}
