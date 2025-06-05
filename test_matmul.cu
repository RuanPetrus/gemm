#include <stdio.h>
#include <chrono>

#include "gemm.cu"

#define TEMP_PATH "/tmp/matmul_"
#define TEST_ASSERT(expr, message, ...) do  { \
if (!(expr)) { \
	fprintf(stderr, message, ##__VA_ARGS__); \
	return false; \
} \
} while(0)

#define LOAD_VAR(x)   TEST_ASSERT(sizeof(x) == fread(&x, 1, sizeof(x), f), "Test bin format is wrong\n");
#define LOAD_ARRAY(x) TEST_ASSERT(sizeof(x) == fread(x, 1, sizeof(x), f), "Test bin format is wrong\n");
#define LOAD_PTR(x, sz) TEST_ASSERT((sz) == fread(x, 1, (sz), f), "Test bin format is wrong\n");


#define TEST_COPY_ARRAY(x, x_exp) cudaMemcpy(x, x_exp, sizeof(x_exp), cudaMemcpyHostToDevice)
#define TEST_COPY_PTR(x, x_exp, sz) cudaMemcpy(x, x_exp, (sz), cudaMemcpyHostToDevice)

#define CLOSE_EPS 1e-4

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

	float x_exp[N * K]; LOAD_ARRAY(x_exp);
	float w_exp[K * M]; LOAD_ARRAY(w_exp);
	float out_exp[N*M]; LOAD_ARRAY(out_exp);
	fclose(f);

	float *x   = (float *) gpu_alloc(sizeof(x_exp)); 
	float *w   = (float *) gpu_alloc(sizeof(w_exp)); 
	float *out = (float *) gpu_alloc(sizeof(out_exp)); 
	TEST_COPY_ARRAY(x, x_exp);
	TEST_COPY_ARRAY(w, w_exp);
	cudaDeviceSynchronize();

	gemm(N, M, K, x, w, out);
	cudaDeviceSynchronize();

	// printf("%d %d %d\n", N, K, M);
	// show_data(x_exp, N*K);
	// show_gpu_data(x, N*K);
	// printf("\n");
	// show_data(w_exp, K*M);
	// show_gpu_data(w, K*M);
	// printf("\n");
	// show_data(out_exp, N*M);
	// show_gpu_data(out, N*M);
	// printf("\n");

	return assert_close(out, out_exp, N*M);
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
