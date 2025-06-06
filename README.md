# GEMM
This is a educational repository where I try to achive cuBLAS like
performance on GEMM
# Results
# MX150 Graphics cards
This is the results on my laptop gpu

Commit-hash: 3f455ed53bd1ce72a8bc68deb04f147e697d1c3d
```sh
python3 torch_matmul.py
./build/test_cublas
--------------------------
cuBlas Implementation:
Average elapsed time: (0.016393) s
performance: (1048.0) GFLOPS.
--------------------------
./build/test_matmul
--------------------------
Gemm Kernel Implementation:
Average elapsed time: (0.024127) s
performance: (712.1) GFLOPS.
--------------------------
```
cuBLAS comparative: 68%  

Commit-hash: 9a69fa707db09a26d9b592cbdf17b6ac6baf162d
```sh
python3 torch_matmul.py
./build/test_cublas
--------------------------
cuBlas Implementation:
Average elapsed time: (0.016384) s
performance: (1048.6) GFLOPS.
--------------------------
./build/test_matmul
--------------------------
Gemm Kernel Implementation:
Average elapsed time: (0.020398) s
performance: (842.2) GFLOPS.
--------------------------
```
cuBLAS comparative: 80%  
