#include <cassert>
#include <cuda_runtime.h>

#define ERROR(message, ...) do { fprintf(stderr, message, ##__VA_ARGS__); abort(); } while(0)

template<const uint BN, const uint BM, const uint BK, 
         const uint TN, const uint TM, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	kernel_gemm(uint N, uint M, uint K, 
				const float *A, const float *B, float *C) 
{
	const uint blrow = blockIdx.y;
	const uint blcol = blockIdx.x; 

	const uint trow  = threadIdx.x / (BM / TM);
	const uint tcol  = threadIdx.x % (BM / TM);

	const uint iArow = threadIdx.x / BK;
	const uint iAcol = threadIdx.x % BK;
	constexpr uint strideA = NUM_THREADS/BK;
	
	const uint iBrow = threadIdx.x / BM;
	const uint iBcol = threadIdx.x % BM;
	constexpr uint strideB = NUM_THREADS/BM;

	__shared__ float SA[BN*BK], SB[BK*BM];

	A += (blrow * BN)*K;
	B += (blcol * BM)*1;
	C += (blrow * BN)*M + (blcol * BM)*1;
	
	float rs[TN*TM] = {0};
	float regA[TN], regB[TM];

	for (uint blit = 0; blit < (K/BK); blit++) {
		// Loading A and B to shared memory
		for (uint offA = 0; offA < BN; offA += strideA) {
			// SA[(iArow+offA)*BK + iAcol] = A[(iArow+offA)*K + iAcol];
			SA[(iAcol)*BN +(iArow+offA)] = A[(iArow+offA)*K + iAcol]; // transposing SA
		}
		for (uint offB = 0; offB < BK; offB += strideB) {
			SB[(iBrow+offB)*BM + iBcol] = B[(iBrow+offB)*M + iBcol];
		}
		__syncthreads();

		// Doing computation
		for (uint k = 0; k < BK; k++) {
			for (int i = 0; i < TN; i++) {
				// regA[i] = SA[(trow*TN+i)*BK +k];
				regA[i] = SA[(k)*BN +(trow*TN+i)];
			}
			for (int i = 0; i < TM; i++) {
				regB[i] = SB[k*BM + (tcol*TM+i)];
			}
			for (uint resN = 0; resN < TN; resN++) {
				for (uint resM = 0; resM < TM; resM++) {
					rs[resN*TM+resM] += regA[resN] *  regB[resM];
				}
			}
		}
		A += BK;
		B += BK*M;
		__syncthreads();
	}
	for (uint resN = 0; resN < TN; resN++) {
		for (uint resM = 0; resM < TM; resM++) {
			C[(trow*TN + resN)*M + (tcol*TM+resM)] = rs[resN*TM+resM];
		}
	}
}

#define CEIL(a, b) (((a)+(b)-1)/ (b))

void gemm(uint N, uint M, uint K, const float *A, const float *B, float *C) 
{
	constexpr uint BN = 128;
	constexpr uint BM = 128;
	constexpr uint BK = 8;
	constexpr uint TN = 8;
	constexpr uint TM = 8;
	constexpr uint NUM_THREADS = (BN*BM) / (TN*TM);

	static_assert(BM % TM == 0);
	static_assert(BN % TM == 0);
	static_assert(NUM_THREADS % BK == 0);
	static_assert(NUM_THREADS % BM == 0);

	assert((N % BN) == 0 && "N must be a multiple of BN\n");
	assert((M % BM) == 0 && "M must be a multiple of BM\n");
	assert((K % BK) == 0 && "K must be a multiple of BK\n");

	dim3 blockDim(NUM_THREADS);
	dim3 gridDim(M/BM, N/BN);
	kernel_gemm
		<BN, BM, BK, TN, TM, NUM_THREADS>
		<<<gridDim, blockDim>>>(N, M, K, A, B, C);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}
