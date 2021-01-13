#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <cuda_runtime.h>

#include "GemmCuda.hpp"
#include "Problem.hpp"
#include "Stov.hpp"

#include "Timer.hpp"

//#define DISPLAY_RESULTS // 計算結果の行列を表示する場合
//#define CPU_PERFORMANCE_RUN // CPUのベンチマーク結果を表示する場合


#define CUDA_HANLDE_ERROR( err ) (handleCudaError(err, __FILE__, __LINE__ ))

namespace
{
	template<typename T>
	void handleCudaError(cudaError_t error, const char filename[],T line)
	{
		if(error != cudaSuccess)
		{
			std::stringstream ss;
			ss << cudaGetErrorString(error) << " in " << filename << " at line "<< line;
			throw std::runtime_error(ss.str());
		}
	}

#ifdef DISPLAY_RESULTS
	// half型のnA*mA行列のmatAを表示する
	void printMatrix(const half* const matA, const std::size_t nA, const std::size_t mA)
	{
		for (auto i = decltype(nA)(0); i < nA; ++i)
		{
			for (auto j = decltype(mA)(0); j < mA; ++j)
			{
				// i行j列の要素の表示
				std::cout << __half2float(matA[i * mA + j]) << "  ";
			}
			std::cout << std::endl;
		}
	}
#endif

	// デリーター（getDevMatrixPtrの戻り値の型を取得するために外に出している）
	struct Deleter
	{
		void operator()(void* const ptr)
			{
				CUDA_HANLDE_ERROR( cudaFree(ptr) );
			}
	};
	
	// デバイス側の行列用メモリの確保
	auto getDevMatrixPtr(const std::size_t nA, const std::size_t mA)
	{
		auto getFltMallocPtr = [](const std::size_t size)
			{
				half* dPtr;
				CUDA_HANLDE_ERROR( cudaMalloc((void**)&dPtr, size) );
				return dPtr;
			};
		return std::unique_ptr<half[], Deleter>(getFltMallocPtr(nA * mA * sizeof(half)));
	}
	
#ifdef CPU_PERFORMANCE_RUN
	// CPU版の行列積の実行
	void cpuProductMatrix(const Gemm::Problem& prob)
	{
		auto matC = prob.benchmark (
			[](const std::size_t, const std::size_t, const std::size_t, const half* const, const half* const)
			{
				// 前処理（ここの処理は時間計測に含まれない）
			},
			[](const std::size_t nA, const std::size_t mB, const std::size_t mA, const half* const matA, const half* const matB, half* const matC)
			{
				// 実際の処理（ここの処理のみが時間計測に含まれる）
				productMatrix(nA, mB, mA, matA, matB, matC);
			},
			[](const std::size_t, const std::size_t, half* const)
			{
				// 後処理（ここの処理は時間計測に含まれない）
			});

#ifdef DISPLAY_RESULTS
		// 結果の行列の表示
		std::cout << "Matrix C = " << std::endl;
		printMatrix(matC.get(), prob.nA, prob.mB);
#endif
	}
#endif

	
	// 行列サイズのKの値を調整
	// TensorCoreを使う場合(arch >= 70) : 0パディングを行わないためそのままkを返す 
	// f16x2を使う場合                  : 0パディングのためにk以上の最小の偶数を返す
	std::size_t adjustK(const std::size_t k){
#ifdef USE_TENSORCORE
		return k;
#else
		return k & 1 ? (k + 1) : k;
#endif
	}
	
	// 最適化版の行列積の実行
	void optProductMatrix(const Gemm::Problem& prob)
	{
		std::unique_ptr<half[], Deleter> dMatA;
		std::unique_ptr<half[], Deleter> dMatB;
		std::unique_ptr<half[], Deleter> dMatC;
		auto matC = prob.benchmark (
			[&dMatA, &dMatB, &dMatC](const std::size_t nA, const std::size_t mB, const std::size_t mA, const half* const matA, const half* const matB)
			{
				// 前処理（ここの処理は時間計測に含まれない）
				// ！！！！！！！！！！！！！ここに最適化版の前処理を書く！！！！！！！！！！！！！
				// デバイスメモリの確保
				const auto adjusted_k = adjustK(mA);
				dMatA = getDevMatrixPtr(nA, adjusted_k);
				dMatB = getDevMatrixPtr(adjusted_k, mB);
				dMatC = getDevMatrixPtr(nA, mB);

				// デバイスメモリにデータを転送
				for (auto i = decltype(nA)(0); i < nA; ++i){
					CUDA_HANLDE_ERROR( cudaMemcpy(dMatA.get() + i * adjusted_k, matA + i * mA, mA * sizeof(half), cudaMemcpyDefault) );
				}
				// 行列matBを行・列優先を変更しながらコピー
				for (auto i = decltype(mA)(0); i < mA; ++i)
				{
					for (auto j = decltype(mB)(0); j < mB; ++j)
					{
						CUDA_HANLDE_ERROR( cudaMemcpy(dMatB.get() + j * adjusted_k + i, matB + i * mB + j, sizeof(half), cudaMemcpyDefault) );
					}
				}
#ifndef USE_TENSORCORE
				// 0パディング
				// cudaMallocは0クリアをしないので必要
				half zero = __float2half(.0f);
				for (auto i = decltype(nA)(0); i < nA; ++i){
					CUDA_HANLDE_ERROR( cudaMemcpy(dMatA.get() + (i+1) * adjusted_k - 1, &zero, sizeof(half), cudaMemcpyDefault) );
				}
				for (auto i = decltype(mB)(0); i < mB; ++i){
					CUDA_HANLDE_ERROR( cudaMemcpy(dMatB.get() + (i+1) * adjusted_k - 1, &zero, sizeof(half), cudaMemcpyDefault) );
				}
#endif
			},
			[&dMatA, &dMatB, &dMatC](const std::size_t nA, const std::size_t mB, const std::size_t mA, const half* const, const half* const, half* const)
			{
				// 実際の処理（ここの処理のみが時間計測に含まれる）
				// ！！！！！！！！！！！！！ここに最適化版の実処理を書く！！！！！！！！！！！！！
#ifdef USE_TENSORCORE
				productMatrixCudaWrap_TC(nA, mB, mA, dMatA.get(), dMatB.get(), dMatC.get());
#else
				productMatrixCudaWrap_x2(nA, mB, adjustK(mA), dMatA.get(), dMatB.get(), dMatC.get());
#endif
			},
			[&dMatC](const std::size_t nC, const std::size_t mC, half* const matC)
			{
				// 後処理（ここの処理は時間計測に含まれない）
				// ！！！！！！！！！！！！！ここに最適化版の後処理を書く！！！！！！！！！！！！！

				// 時間計測対象のカーネルのエラーを処理する
				CUDA_HANLDE_ERROR( cudaGetLastError() );
				CUDA_HANLDE_ERROR( cudaMemcpy(matC, dMatC.get(), nC * mC * sizeof(half), cudaMemcpyDefault) );
			});

#ifdef DISPLAY_RESULTS
		// 結果の行列の表示
		std::cout << "Matrix C = " << std::endl;
		printMatrix(matC.get(), prob.nA, prob.mB);
#endif
	}
	
}
	
int main(int argc, const char* argv[])
{
	// 行列のサイズに指定がある場合は取得
	// なければ1*1行列にする
	// 1つ目の行列
	const auto nA = argc >= 2 ? stov<std::size_t>(argv[1]) : std::size_t(1);
	const auto mA = argc >= 3 ? stov<std::size_t>(argv[2]) : std::size_t(1);
	// 2つ目の行列
	const auto nB = mA;
	const auto mB = argc >= 4 ? stov<std::size_t>(argv[3]) : std::size_t(1);
	// 結果の行列
	const auto nC = nA;
	const auto mC = mB;

	// 計算する行列サイズの表示
	std::cout << "Matrix A size: "
	          << nA << "*" << mA
	          << std::endl;
	std::cout << "Matrix B size: "
	          << nB << "*" << mB
	          << std::endl;
	std::cout << "Matrix C size: "
	          << nC << "*" << mC
	          << std::endl;
	
	// 問題の作成
	Gemm::Problem prob(nA, mB, mA);

#ifdef DISPLAY_RESULTS
	// 行列の表示
	// 1つ目の行列
	std::cout << "Matrix A = " << std::endl;
	printMatrix(prob.matA.get(), nA, mA);
	// 2つ目の行列
	std::cout << "Matrix B = " << std::endl;
	printMatrix(prob.matB.get(), nB, mB);
#endif

#ifdef CPU_PERFORMANCE_RUN
	// CPU版の呼び出し
	std::cout << "CPU:" << std::endl;
	cpuProductMatrix(prob);
#endif
	
	try
	{
		// 最適化版の呼び出し
		std::cout << "OPT:" << std::endl;
		optProductMatrix(prob);
	}
	catch(std::exception& e)
	{
		std::cerr<<e.what()<<std::endl;
		return 1;
	}
	
	return 0;
}
