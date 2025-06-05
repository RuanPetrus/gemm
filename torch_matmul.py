import torch

TEMP_PATH = "/tmp/matmul_"

def test_matmul():
    N = 1024
    K = 1024
    M = 1024

    x = torch.randn(N, K, dtype=torch.float32)
    w = torch.randn(K, M, dtype=torch.float32)
    c = x @ w

    with open(TEMP_PATH + "matmul.bin", "wb") as f:
        f.write(N.to_bytes(4, byteorder='little', signed=True))
        f.write(K.to_bytes(4, byteorder='little', signed=True))
        f.write(M.to_bytes(4, byteorder='little', signed=True))
        x.numpy().tofile(f)
        w.numpy().tofile(f)
        c.numpy().tofile(f)

def main():
    test_matmul()

if __name__ == "__main__":
    main()
