
CFLAGS="-Wall,-Wextra"
# NVCC_FLAGS=-std=c++17 -O2 --ptxas-options="-v "
NVCC_FLAGS=-std=c++17 -O3 -arch=native

.PHONY: all test clean
all: test build/device_info

build/test_matmul: test_matmul.cu gemm.cu
	@mkdir -p $(dir $@)
	nvcc -Xcompiler $(CFLAGS) $(NVCC_FLAGS) test_matmul.cu -o $@

build/test_cublas: test_cublas.cu
	@mkdir -p $(dir $@)
	nvcc -Xcompiler $(CFLAGS) $(NVCC_FLAGS) test_cublas.cu -lcublas -o $@

test: build/test_matmul build/test_cublas
	python3 torch_matmul.py
	./build/test_cublas
	./build/test_matmul

build/device_info: device_info.cu
	@mkdir -p $(dir $@)
	nvcc -Xcompiler $(CFLAGS) $(NVCC_FLAGS) device_info.cu -o $@

clean:
	rm -rf build
