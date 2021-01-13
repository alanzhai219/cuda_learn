#ifndef HEADER_GEMM_CPU_NAIVE
#define HEADER_GEMM_CPU_NAIVE
#include <cuda_fp16.h>

// half型のnC*mAの行列matAとmA*mCの行列matBの積を求めて行列matCに格納する
void productMatrix(const std::size_t nC, const std::size_t mC, const std::size_t mA, const half* const matA, const half* const matB, half* const matC)
{
	for (auto i = decltype(nC)(0); i < nC; ++i)
	{
		for (auto j = decltype(mC)(0); j < mC; ++j)
		{
			half cIJ = __float2half(0.f);
			for (auto k = decltype(mA)(0); k < mA; ++k)
			{
				// sum_k (A_ik * B_kj) = C_ij を計算
				// GPU版ではfmaを用いているので丸めるのは和計算のあと
				cIJ = __float2half(__half2float(matA[i * mA + k])*__half2float(matB[k * mC + j]) + __half2float(cIJ));
			}
			matC[i * mC + j] = __float2half(cIJ);
		}
	}
}

#endif
