#include <cassert>
#include <cuda_runtime.h>

#define ERROR(message, ...) do { fprintf(stderr, message, ##__VA_ARGS__); abort(); } while(0)

template<const int BN, const int BM, const int BK>
__global__ void kernel_gemm(int N, int M, int K, 
							const float *A, const float *B, float *C) 
{
	const int blrow = blockIdx.y;
	const int blcol = blockIdx.x;
	const int trow = threadIdx.y;
	const int tcol = threadIdx.x;

	__shared__ float SA[BN*BK], SB[BK*BM];

	A += (blrow * BN)*K;
	B += (blcol * BM)*1;
	C += (blrow * BN)*M + (blcol * BM)*1;

	float sum = 0;
	for (int blit = 0; blit < (K/BK); blit++) {
		// Loading A and B to shared memory
		__syncthreads();
		SA[trow*BK + tcol] = A[trow*K + tcol];
		SB[trow*BM + tcol] = B[trow*M + tcol];

		__syncthreads();

		// Doing computation
		for (int k = 0; k < BK; k++) {
			sum += SA[trow*BK + k] * SB[k*BM + tcol];
		}
		A += BK;
		B += BK*M;
	}
	C[trow*M + tcol] = sum;
}

#define CEIL(a, b) (((a)+(b)-1)/ (b))

void gemm(int N, int M, int K, const float *A, const float *B, float *C) 
{
	constexpr int BN = 32;
	constexpr int BM = 32;
	constexpr int BK = 32;

	assert((N % BN) == 0 && "N must be a multiple of BN\n");
	assert((M % BM) == 0 && "M must be a multiple of BM\n");
	assert((K % BK) == 0 && "K must be a multiple of BK\n");

	static_assert(BK == BN, "For now BK must be equal to BN\n");
	static_assert(BK == BM, "For now BM must be equal to BN\n");

	dim3 blockDim(BM, BN);
	dim3 gridDim(M/BM, N/BN);
	kernel_gemm
		<BN, BM, BK>
		<<<gridDim, blockDim>>>(N, M, K, A, B, C);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}
