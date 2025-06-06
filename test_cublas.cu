#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TEMP_PATH "/tmp/matmul_"
#define TEST_ASSERT(expr, message, ...) do  { \
if (!(expr)) { \
	fprintf(stderr, message, ##__VA_ARGS__); \
	return false; \
} \
} while(0)

#define CHECK_CUBLAS(call) TEST_ASSERT((call) == CUBLAS_STATUS_SUCCESS, "cuBLAS error\n");

#define LOAD_VAR(x)   TEST_ASSERT(sizeof(x) == fread(&x, 1, sizeof(x), f), "Test bin format is wrong\n");
#define LOAD_ARRAY(x) TEST_ASSERT(sizeof(x) == fread(x, 1, sizeof(x), f), "Test bin format is wrong\n");
#define LOAD_PTR(x, sz) TEST_ASSERT((sz) == fread(x, 1, (sz), f), "Test bin format is wrong\n");


#define TEST_COPY_ARRAY(x, x_exp) cudaMemcpy(x, x_exp, sizeof(x_exp), cudaMemcpyHostToDevice)
#define TEST_COPY_PTR(x, x_exp, sz) cudaMemcpy(x, x_exp, (sz), cudaMemcpyHostToDevice)

#define CLOSE_EPS 1e-3

char *gpu_alloc(size_t n) 
{
	char *ptr; cudaMalloc(&ptr, n);
	return ptr;
}

float *gpu_alloc_float(size_t n) 
{
	return (float*)gpu_alloc(n * sizeof(float));
}

char *cpu_alloc(size_t n) 
{
	return (char*) malloc(n);
}

float *cpu_alloc_float(size_t n) 
{
	return (float*)cpu_alloc(n * sizeof(float));
}

bool assert_close(float *gpu_data, float *exp, int n, float eps = CLOSE_EPS)
{
	float *data = cpu_alloc_float(n);
	cudaMemcpy(data, gpu_data, sizeof(float) * n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		float diff = abs(data[i] - exp[i]);
		TEST_ASSERT(diff < eps, "Number are not close (i, diff) = (%d, %f)", i, diff);
	}
	return true;
}

bool show_gpu_data(float *gpu_data, int n) 
{
	float *data = cpu_alloc_float(n);
	cudaMemcpy(data, gpu_data, sizeof(float) * n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		printf("%.4f ", data[i]);
	}
	printf("\n");
	return true;
}

bool show_data(float *data, int n) 
{
	for (int i = 0; i < n; i++) {
		printf("%.4f ", data[i]);
	}
	printf("\n");
	return true;
}

bool test_matmul()
{
	FILE *f = fopen(TEMP_PATH"matmul.bin", "rb");
	TEST_ASSERT(f != NULL, "Could not open test_matmul bin\n");

	int N, K, M;
	LOAD_VAR(N); LOAD_VAR(K); LOAD_VAR(M);

	float *x_exp = cpu_alloc_float(N*K);   LOAD_PTR(x_exp, N*K*sizeof(float));
	float *w_exp = cpu_alloc_float(K*M);   LOAD_PTR(w_exp, K*M*sizeof(float));
	float *out_exp = cpu_alloc_float(N*M); LOAD_PTR(out_exp, N*M*sizeof(float));
	fclose(f);

	float *x   = (float *) gpu_alloc(N*K*sizeof(float)); 
	float *w   = (float *) gpu_alloc(K*M*sizeof(float)); 
	float *out = (float *) gpu_alloc(N*M*sizeof(float)); 

	TEST_COPY_PTR(x, x_exp, N*K*sizeof(float));
	TEST_COPY_PTR(w, w_exp, K*M*sizeof(float));
	cudaDeviceSynchronize();

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

	/*
	Cublas works in collum major order, so is the same as trasnposing
	the matrix, so we are going to calculate, C^T
	C^T = B^T @ A^T
	*/

	CHECK_CUBLAS(cublasSgemm(
		handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		N, M, K,          // Note the switched N and M for column-major
		&alpha,
		w, M,
		x, K,
		&beta,
		out, M));

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("cudaGetErrorString(err) = %s\n", cudaGetErrorString(err));
		abort();
	}
	if (!assert_close(out, out_exp, N*M)) return false;

	// Warming cache
	for (int z = 0; z < 10; z++) {
		CHECK_CUBLAS(cublasSgemm(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			N, M, K,          // Note the switched N and M for column-major
			&alpha,
			w, M,
			x, K,
			&beta,
			out, M));
		cudaDeviceSynchronize();
	}
	const int T = 30;
	double sum_gflops = 0;
	for (int z = 0; z < T; z++) {
		auto start = std::chrono::steady_clock::now();
		CHECK_CUBLAS(cublasSgemm(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			N, M, K,          // Note the switched N and M for column-major
			&alpha,
			w, M,
			x, K,
			&beta,
			out, M));
		cudaDeviceSynchronize();
		auto stop = std::chrono::steady_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
		long double duration_nano = duration.count();
		double gflop = 2*(double)N*K*M;
		double gflops = gflop / duration_nano;
		sum_gflops += gflops;
	}
    cublasDestroy(handle);
	printf("Cublas MatMul Gflops = %lf\n", sum_gflops / T);
	return true;
}

int main()
{
	int errors = 0;
	errors += !test_matmul();

	if (errors > 0) {
		fprintf(stderr, "Tests failed with %d errors\n", errors);
		return 1;
	}

	fprintf(stdout, "SUCESS\n");

	return 0;
}
