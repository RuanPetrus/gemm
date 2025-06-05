
CFLAGS="-Wall,-Wextra"
NVCC_FLAGS=-std=c++17

.PHONY: all test clean
all: test

test_matmul: test_matmul.cu gemm.cu
	nvcc -Xcompiler $(CFLAGS) $(NVCC_FLAGS) test_matmul.cu -o test_matmul

test: test_matmul
	python3 torch_matmul.py
	./test_matmul

clean:
	rm test_matmul
