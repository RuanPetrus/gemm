#include <stdio.h>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUBLAS(call) \
    if ((call) != CUBLAS_STATUS_SUCCESS) { \
		printf("cuBLAS error\n"); \
        return EXIT_FAILURE; \
    }

struct Tensor 
{
	float *data;
	float *cpu_data;
	int sz;
};

Tensor tensor_new(int n) 
{
	Tensor t;
	t.sz = n;
	t.cpu_data = (float*) malloc(n * sizeof(float));
	cudaMalloc(&t.data, n * sizeof(float));
	
	return t;
}

void tensor_copy_gpu_to_cpu(Tensor t) 
{
	cudaDeviceSynchronize();
	cudaMemcpy(t.cpu_data, t.data, t.sz * sizeof(float), cudaMemcpyDeviceToHost);
}

void tensor_copy_cpu_to_gpu(Tensor t) 
{
	cudaDeviceSynchronize();
	cudaMemcpy(t.data, t.cpu_data, t.sz * sizeof(float), cudaMemcpyHostToDevice);
}

float random_normal_distribution_float()
{
	static std::random_device random_device{};
	static std::mt19937 random_generator{random_device()};
	static std::normal_distribution random_normal_distribution{0.0f, 1.0f};
	return random_normal_distribution(random_generator);
}

Tensor tensor_randn(int sz)
{
	Tensor t = tensor_new(sz);
	for (int i = 0; i < t.sz; i++) {
		t.cpu_data[i] = random_normal_distribution_float();
	}
	tensor_copy_cpu_to_gpu(t);
	return t;
}

void tensor_show(Tensor t)
{
	tensor_copy_gpu_to_cpu(t);
	printf("Tensor (%d):\n", t.sz);
	for (int i = 0; i < t.sz; i++) {
		printf("%.4f ", t.cpu_data[i]);
	}
	printf("\n");
}

void tensor_sum(Tensor t)
{
	tensor_copy_gpu_to_cpu(t);
	float sum = 0;
	for (int i = 0; i < t.sz; i++) {
		sum += t.cpu_data[i];
	}
	printf("SUM: %.4f\n", sum);
}

int int_ceil(int a, int b)
{
	return (a+b-1) / b;
}

const int N = 4096;
int main() 
{
	Tensor a = tensor_randn(N*N);
	Tensor b = tensor_randn(N*N);
	Tensor c = tensor_new(N*N);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

	for (int i = 0; i < 100; i++) {
		auto start = std::chrono::steady_clock::now();

		// C = alpha * A * B + beta * C
		// cuBLAS uses column-major storage, so we switch the order
		CHECK_CUBLAS(cublasSgemm(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			N, N, N,          // Note the switched N and M for column-major
			&alpha,
			b.data, N,
			a.data, N,
			&beta,
			c.data, N));

		cudaDeviceSynchronize();
		auto stop = std::chrono::steady_clock::now();

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("cudaGetErrorString(err) = %s\n", cudaGetErrorString(err));
			abort();
		}
		// tensor_sum(c);
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
		long double duration_nano = duration.count();
		double gflops = (double) N * N * 2* N / duration_nano;

		// printf("Time(s)= %lf\n", (double) duration_nano / 1e9);
		printf("Gflops = %lf\n", gflops);
	}
    cublasDestroy(handle);
	return 0;
}
