#include <stdio.h>
#include <random>
#include <chrono>
#include <assert.h>

#define KEY(i2, i1, d2, d1) (i2 * d1 + i1)
#define BS 32

__global__ void kernel_mat_mul(
		float *c, float *a, float *b, 
		int N, int M, int K) 
{
	__shared__ float tileA[BS * BS];
    __shared__ float tileB[BS * BS];

	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	a += (by * BS) * K;
	b += (bx * BS);

	int iy = by * BS + ty;
	int ix = bx * BS + tx;

	if (iy < N && ix < M) {
		float sum = 0;
		for (int t = 0; t < K/BS; t++) {
			tileA[ty * BS + tx] = a[ty * K + tx];
			tileB[ty * BS + tx] = b[ty * M + tx];

			__syncthreads();

			for (int k = 0; k < BS; k++) {
				sum += tileA[ty * BS + k] * tileB[k * BS + tx];
			}
			a += BS;
			b += BS*K;
			__syncthreads();
		}
		c[KEY(iy, ix, N, M)] = sum;
	}
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

const int N = 2048;
float expected[N*N];

int main() 
{
	dim3 threads_per_block(BS, BS);
	dim3 num_blocks(
			int_ceil(N, BS), 
			int_ceil(N, BS)
	);

	// dim3 threads_per_block(BLOCKSIZE*BLOCKSIZE);
	// dim3 num_blocks(
	// 		int_ceil(N, BLOCKSIZE), 
	// 		int_ceil(N, BLOCKSIZE)
	// );

	Tensor a = tensor_new(N*N);
	Tensor b = tensor_new(N*N);
	Tensor c = tensor_new(N*N);

	FILE *f = fopen("/tmp/arr", "rb");
	assert(f != NULL);
	assert(fread(a.cpu_data, 1, sizeof(float) * a.sz, f) 
			== sizeof(float) * a.sz);
	assert(fread(b.cpu_data, 1, sizeof(float) * b.sz, f) 
			== sizeof(float) * b.sz);
	assert(fread(expected, 1, sizeof(float) * c.sz, f) 
			== sizeof(float) * c.sz);

	tensor_copy_cpu_to_gpu(a);
	tensor_copy_cpu_to_gpu(b);
	cudaDeviceSynchronize();
	fclose(f);

	kernel_mat_mul<<<num_blocks, threads_per_block>>>(
		c.data, a.data, b.data,
		N, N, N
	);

	cudaDeviceSynchronize();
	tensor_copy_gpu_to_cpu(c);

	for (int i = 0; i < c.sz; i++) {
		float diff = abs(c.cpu_data[i] - expected[i]);
		if (diff > 1e-3) {
			printf("(%d, %.5f, %.5f, %.5f)\n", i, diff, c.cpu_data[i], expected[i]);
			return 1;
		}
	}

	for (int i = 0; i < 100; i++) {
		auto start = std::chrono::steady_clock::now();

		kernel_mat_mul<<<num_blocks, threads_per_block>>>(
			c.data, a.data, b.data,
			N, N, N
		);

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
	return 0;
}
