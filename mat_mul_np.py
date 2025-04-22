import numpy as np
import time

N = 2048
if __name__ == "__main__":
    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    c = a@b
    with open("/tmp/arr", "wb") as f:
        a.tofile(f)
        b.tofile(f)
        c.tofile(f)

    for i in range(100):
        st = time.monotonic()
        c = a@b
        et = time.monotonic()
        s = et -st
        flop = N * N * 2 *N
        #print(f"Time(s) = {s}")
        print(f"Gflops = {flop / s / 1e9}")
    
