
CFLAGS="-Wall,-Wextra"
# NVCC_FLAGS=-std=c++17 -O2 --ptxas-options="-v "
NVCC_FLAGS=-std=c++17 -O3 -arch=native

.PHONY: all test clean
all: test device_info

test_matmul: test_matmul.cu gemm.cu
	nvcc -Xcompiler $(CFLAGS) $(NVCC_FLAGS) test_matmul.cu -o $@

test: test_matmul
	python3 torch_matmul.py
	./test_matmul

device_info: device_info.cu
	nvcc -Xcompiler $(CFLAGS) $(NVCC_FLAGS) device_info.cu -o $@

clean:
	rm test_matmul
