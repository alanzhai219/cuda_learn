#include "GemmCuda.hpp"
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>

#define PTX

// 行列積の計算 GPU FP16 版
__global__
void productMatrixCuda_x1(const std::size_t dNA, const std::size_t dMB, const std::size_t dMA, const half* __restrict__ const dMatA, const half* __restrict__ const dMatB, half* const dMatC)
{
	// スレッド番号の計算
	const auto idy = blockIdx.y * blockDim.y + threadIdx.y;
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	// 範囲外なら終了
	if ((idx >= dNA) || (idy >= dMB)) return;
	
	union {
		half h;
		unsigned short us;
	} cIJ;
	cIJ.h = __float2half(0.f);
	const half* pA = dMatA + idx * dMA;
	const half* pB = dMatB + idy * dMA;

	for (auto k = decltype(idx)(0); k < dMA; ++k)
	{
#ifdef PTX
		asm(
				R"(
{
	.reg .b16 %ma<2>;
	ld.global.nc.b16 %ma0, [%1];
	ld.global.nc.b16 %ma1, [%2];
	fma.rn.f16 %0, %ma0, %ma1, %0;
}
)":"+h"(cIJ.us):"l"(pA + k),"l"(pB + k)
				);
#else
		cIJ.h = __hfma(__ldg(pA + k) , __ldg(pB + k), cIJ.h);
#endif
	}
	dMatC[idx * dMB + idy] = cIJ.h;
}

// nvccとclang++を分けるためのラッパ
void productMatrixCudaWrap_x1(const std::size_t nA, const std::size_t mB, const std::size_t mA, const half* const dMatA, const half* const dMatB, half* const dMatC)
{
	// ここで行列のn, m方向のブロック内のスレッド数を決める
	// ワープ内のスレッド数に合わせて特にブロックサイズのx方向（行列のn方向）は32の倍数に設定した方が良い
	// ここでは、ワープ内スレッド数に合わせて32スレッドでブロックサイズを決めている（理由はないがy方向も同じにした）
	const dim3 block (32, 32, 1);
	const dim3 grid((nA + block.x - 1) / block.x, (mB + block.y - 1) / block.y, 1);
	productMatrixCuda_x1<<<grid, block>>>(nA, mB, mA, dMatA, dMatB, dMatC);
	cudaDeviceSynchronize();
}


// 行列積の計算 GPU FP16x2 版
// dMAが偶数だった場合のカーネル
__global__
void productMatrixCuda_x2_even(const std::size_t dNA, const std::size_t dMB, const std::size_t dMA, const half* __restrict__ const dMatA, const half* __restrict__ const dMatB, half* const dMatC)
{
	// スレッド番号の計算
	const auto idy = blockIdx.y * blockDim.y + threadIdx.y;
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int dMA_div_2 = dMA/2;
	// 範囲外なら終了
	if ((idx >= dNA) || (idy >= dMB)) return;
	
	union {
		unsigned int ui;
		__half2 f2;
	} cIJ_x2;
	cIJ_x2.f2 = __float2half2_rn(0.f);

	const half2 *pA_x2 = (const half2*)( dMatA + idx * dMA );
	const half2 *pB_x2 = (const half2*)( dMatB + idy * dMA );

	for (auto k = decltype(idy)(0); k < dMA_div_2; ++k)
	{
#ifdef PTX
		asm(
			R"(
{
	.reg .b32 %ma<2>;
	ld.global.nc.b32 %ma0, [%1];
	ld.global.nc.b32 %ma1, [%2];
	fma.rn.f16x2 %0, %ma0, %ma1, %0;
}
)":"+r"(cIJ_x2.ui):"l"(pA_x2 + k),"l"(pB_x2 + k)
		);
#else
		cIJ_x2.f2 = __hfma2( __ldg(pA_x2 + k), __ldg(pB_x2 + k), cIJ_x2.f2);
#endif
	}

	dMatC[idx * dMB + idy] = __hadd(__high2half(cIJ_x2.f2) , __low2half(cIJ_x2.f2));
}

// nvccとclang++を分けるためのラッパ
void productMatrixCudaWrap_x2(const std::size_t nA, const std::size_t mB, const std::size_t mA, const half* const dMatA, const half* const dMatB, half* const dMatC)
{
	// ここで行列のn, m方向のブロック内のスレッド数を決める
	// ワープ内のスレッド数に合わせて特にブロックサイズのx方向（行列のn方向）は32の倍数に設定した方が良い
	// ここでは、ワープ内スレッド数に合わせて32スレッドでブロックサイズを決めている（理由はないがy方向も同じにした）
	const dim3 block (32, 32);
	const dim3 grid((nA + block.x - 1) / block.x, (mB + block.y - 1) / block.y, 1);

	productMatrixCuda_x2_even<<<grid, block>>>(nA, mB, mA, dMatA, dMatB, dMatC);
	cudaDeviceSynchronize();
}

#ifdef USE_TENSORCORE

using fragment_size_t = int;

constexpr fragment_size_t WMMA_M = 16;
constexpr fragment_size_t WMMA_N = 16;
constexpr fragment_size_t WMMA_K = 16;


__global__
void productMatrixCuda_TC16(const std::size_t dNA, const std::size_t dMB, const std::size_t dMA, const half* __restrict__ const dMatA, const half* __restrict__ const dMatB, half* __restrict__ const dMatC)
{
	const auto warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	const auto warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;

	nvcuda::wmma::fill_fragment(acc_frag, __float2half(.0f));

	const auto a_row = warpM * WMMA_M;
	const auto b_col = warpN * WMMA_N;
	if(a_row >= dNA || b_col >= dMB)return;

	for(auto i = decltype(dMA)(0); i < dMA; i+=WMMA_K)
	{
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
		const half *a_ptr = dMatA + i + a_row * dMA;
		const half *b_ptr = dMatB + i + b_col * dMA;
		nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, dMA);
		nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, dMA);

		nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	}
	const auto c_row = warpM * WMMA_M;
	const auto c_col = warpN * WMMA_N;

	if((c_row < dNA) && (c_col < dMB))
	{
		nvcuda::wmma::store_matrix_sync(dMatC + c_col + c_row * dMB, acc_frag, dMB,  nvcuda::wmma::mem_row_major);
	}
}

#include "FragmentFunctions.cuh"

namespace
{
__device__ 
std::size_t sizeMin(const std::size_t a, const std::size_t b)
{
	return a > b ? b : a;
}
}

__global__
void productMatrixCuda_TC(const std::size_t dNA, const std::size_t dMB, const std::size_t dMA, const half* __restrict__ const dMatA, const half* __restrict__ const dMatB, half* __restrict__ const dMatC)
{
	const auto warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	const auto warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	const auto a_row = warpM * WMMA_M;
	const auto b_col = warpN * WMMA_N;
	if(a_row >= dNA || b_col >= dMB)return;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;

	nvcuda::wmma::fill_fragment(acc_frag, __float2half(.0f));

	const auto frag_real_rows = sizeMin(16, dNA - a_row);
	const auto frag_real_cols = sizeMin(16, dMB - b_col);

	// K方向のループ
	// WMMA_K要素ずつに区切って読み込む
	auto k = decltype(dMA)(0);
	for(; k < dMA - (WMMA_K - 1); k+=WMMA_K)
	{
		load_irregular_matrix(a_frag, dMatA + k + a_row * dMA, dMA, frag_real_rows, WMMA_K);
		load_irregular_matrix(b_frag, dMatB + k + b_col * dMA, dMA, WMMA_K, frag_real_cols);
		nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	}

	// 端数処理
	// dMAがWMMA_Kの倍数なら前のループでk == dMAとなっている
	if(dMA - k){
		load_irregular_matrix(a_frag, dMatA + k + a_row * dMA, dMA, frag_real_rows, dMA - k);
		load_irregular_matrix(b_frag, dMatB + k + b_col * dMA, dMA, dMA - k, frag_real_cols);
		nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	}

	const auto c_row = warpM * WMMA_M;
	const auto c_col = warpN * WMMA_N;

	if((c_row < dNA) && (c_col < dMB))
	{
		half *c_ptr = dMatC + c_col + c_row * dMB;
		if((dMB & 15 == 0) && (frag_real_cols == 16) && (frag_real_rows == 16))
		{
			nvcuda::wmma::store_matrix_sync(c_ptr, acc_frag, dMB,  nvcuda::wmma::mem_row_major);
		}
		else
		{
			store_irregular_matrix(c_ptr, acc_frag, dMB, frag_real_rows, frag_real_cols);
		}
	}
}

// nvccとclang++を分けるためのラッパ
void productMatrixCudaWrap_TC(const std::size_t nA, const std::size_t mB, const std::size_t mA, const half* const dMatA, const half* const dMatB,half* const  dMatC)
{
	// 1ブロック1ワープ32スレッド
	constexpr std::size_t warpSize = 32;
	const dim3 block_dim(warpSize);
	// A,Bの行列サイズが16の倍数の場合はShared Memoryを使わない関数を呼ぶ
	if((nA % 16 == 0) && (mB % 16 == 0) && (mA % 16 == 0))
	{
		// 1グリッドWMMA_M x WMMA_N fragment
		const dim3 grid_dim(nA / WMMA_M,mB / WMMA_N);

		productMatrixCuda_TC16<<<grid_dim, block_dim>>>(nA, mB, mA, dMatA, dMatB, dMatC);
	}
	else
	{
		const dim3 grid_dim((nA + (WMMA_M * block_dim.x / warpSize - 1)) / (WMMA_M * block_dim.x / warpSize),
				(mB + WMMA_N * block_dim.y - 1) / (WMMA_N * block_dim.y));

		productMatrixCuda_TC<<<grid_dim, block_dim>>>(nA, mB, mA, dMatA, dMatB, dMatC);
	}
	cudaDeviceSynchronize();
}
#endif
