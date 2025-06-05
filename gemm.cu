#include <cassert>
#include <cuda_runtime.h>

#define ERROR(message, ...) do { fprintf(stderr, message, ##__VA_ARGS__); abort(); } while(0)

__global__ void kernel_gemm(int N, int M, int K, const float *A, const float *B, float *C) 
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < M) {
		float sum = 0;
		for (int kit = 0; kit < K; kit++) {
			sum += A[row*K + kit] * B[kit*M + col];
		}
		C[row*M + col] = sum;
	}
}

#define CEIL(a, b) (((a)+(b)-1)/ (b))

void gemm(int N, int M, int K, const float *A, const float *B, float *C) 
{
	dim3 blockDim(32, 32);
	dim3 gridDim(CEIL(M, 32), CEIL(N, 32));
	kernel_gemm<<<gridDim, blockDim>>>(N, M, K, A, B, C);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
	}
}
