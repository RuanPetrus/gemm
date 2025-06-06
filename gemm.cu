#include <cassert>
#include <cuda_runtime.h>

#define ERROR(message, ...) do { fprintf(stderr, message, ##__VA_ARGS__); abort(); } while(0)

#define WARP_SIZE 32

template<const uint BN, const uint BM, const uint BK,
		 const uint WN, const uint WM, const uint WMITER,
         const uint TN, const uint TM, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	kernel_gemm(uint N, uint M, uint K, 
				const float *A, const float *B, float *C) 
{
	const uint blrow = blockIdx.y;
	const uint blcol = blockIdx.x; 

	const uint iArow = threadIdx.x / (BK / 4);
	const uint iAcol = threadIdx.x % (BK / 4);
	constexpr uint strideA = NUM_THREADS / (BK / 4);
	
	const uint iBrow = threadIdx.x / (BM / 4);
	const uint iBcol = threadIdx.x % (BM / 4);
	constexpr uint strideB = NUM_THREADS / (BM / 4);

	const uint widx = threadIdx.x / WARP_SIZE;
	const uint wrow = widx / (BM / WM);
	const uint wcol = widx % (BM / WM);

	constexpr uint WNITER = (WN*WM) / (WARP_SIZE*TN*TM*WMITER);
	constexpr uint SUBWN = WN / WNITER;
	constexpr uint SUBWM = WM / WMITER;

	const uint tidx = threadIdx.x % WARP_SIZE;
	const uint trow = tidx / (SUBWM / TM);
	const uint tcol = tidx % (SUBWM / TM);

	__shared__ float SA[BN*BK], SB[BK*BM];

	A += (blrow * BN)*K;
	B += (blcol * BM)*1;
	C += (blrow * BN + wrow * WN)*M + (blcol * BM + wcol *WM)*1;
	
	float rs[WNITER*WMITER*TN*TM] = {0}; // [WNITER][WMITER][TN][TM]
	float regA[WNITER*TN], regB[WMITER*TM];

	for (uint blit = 0; blit < (K/BK); blit++) {
		// Loading A and B to shared memory
		for (uint offA = 0; offA < BN; offA += strideA) {
			float4 tmp = reinterpret_cast<const float4 *>(&A[(iArow+offA)*K + (iAcol)*4])[0];
			// Transposing SA
			SA[(iAcol * 4 + 0)*BN +(iArow+offA)] = tmp.x;
			SA[(iAcol * 4 + 1)*BN +(iArow+offA)] = tmp.y;
			SA[(iAcol * 4 + 2)*BN +(iArow+offA)] = tmp.z;
			SA[(iAcol * 4 + 3)*BN +(iArow+offA)] = tmp.w;
		}
		for (uint offB = 0; offB < BK; offB += strideB) {
			float4 tmp = reinterpret_cast<const float4 *>(&B[(iBrow+offB)*M + iBcol*4])[0];
			SB[(iBrow+offB)*BM + iBcol*4 + 0] = tmp.x;
			SB[(iBrow+offB)*BM + iBcol*4 + 1] = tmp.y;
			SB[(iBrow+offB)*BM + iBcol*4 + 2] = tmp.z;
			SB[(iBrow+offB)*BM + iBcol*4 + 3] = tmp.w;
		}
		__syncthreads();

		// Doing computation
		for (uint k = 0; k < BK; k++) {
			for (uint iSubN = 0; iSubN < WNITER; iSubN++) {
				for (uint rn = 0; rn < TN; rn++) {
					regA[iSubN*TN + rn] = SA[k*BN + (wrow*WN + iSubN*SUBWN + trow*TN + rn)];
				}
			}
			for (uint iSubM = 0; iSubM < WMITER; iSubM++) {
				for (uint rm = 0; rm < TM; rm++) {
					regB[iSubM*TM + rm] = SB[k*BM + (wcol*WM + iSubM*SUBWM + tcol*TM + rm)];
				}
			}

			for (uint iSubN = 0; iSubN < WNITER; iSubN++) {
				for (uint iSubM = 0; iSubM < WMITER; iSubM++) {
					for (uint rn = 0; rn < TN; rn++) {
						for (uint rm = 0; rm < TM; rm++) {
							rs[iSubN*WMITER*TN*TM + iSubM*TN*TM + rn*TM + rm] += 
								regA[iSubN*TN + rn] * regB[iSubM*TM + rm];
						}
					}
				}
			}
		}
		A += BK;
		B += BK*M;
		__syncthreads();
	}
	for (uint iSubN = 0; iSubN < WNITER; iSubN++) {
		for (uint iSubM = 0; iSubM < WMITER; iSubM++) {
			for (uint rn = 0; rn < TN; rn++) {
				for (uint rm = 0; rm < TM; rm += 4) {
					float4 tmp = reinterpret_cast<float4 *>(&rs[
						iSubN*WMITER*TN*TM + iSubM*TN*TM + rn*TM + rm]
					)[0];
					reinterpret_cast<float4 *>
						(&C[(iSubN*SUBWN + trow*TN + rn)*M + (iSubM*SUBWM + tcol*TM + rm)])[0] = tmp;
				}
			}
		}
	}
}

#define CEIL(a, b) (((a)+(b)-1)/ (b))

void gemm(uint N, uint M, uint K, const float *A, const float *B, float *C) 
{
	constexpr uint NUM_THREADS = 128;
	constexpr uint BN = 128;
	constexpr uint BM = 64;
	constexpr uint BK = 16;
	constexpr uint WN = 64;
	constexpr uint WM = 32;
	constexpr uint WMITER = 1;
	constexpr uint TN = 4;
	constexpr uint TM = 4;

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
		<BN, BM, BK, 
		WN, WM, WMITER,
		TN, TM, NUM_THREADS>
		<<<gridDim, blockDim>>>(N, M, K, A, B, C);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}
