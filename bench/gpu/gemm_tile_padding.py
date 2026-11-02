"""Time of the SCF Vxc pass's GEMM, ao (n x 4096) . aow^T (4096 x n), against a grid block's number of AOs n, as
cupy (cuBLAS) runs it (node1). If the time follows the kernel's padded tiles rather than n^2, the padding costs FP64
time. Prints the median time of --repeat calls per n, the useful flop rate (2 n^2 4096 per call) and the rate
counting 128-row by 64-column tiles (volta_dgemm_128x64_tn, the kernel Nsight Compute shows for these GEMMs).
Usage: gemm_tile_padding.py [--ngrids 4096] [--repeat 25]"""
import argparse
import cupy

ap = argparse.ArgumentParser()
ap.add_argument('--ngrids', type=int, default=4096)
ap.add_argument('--repeat', type=int, default=25)
args = ap.parse_args()
g = args.ngrids
print(f"{'n':>4s} {'ms':>7s} {'useful GFLOP/s':>14s} {'GFLOP/s counting 128x64 tiles':>30s}")
for n in (96, 128, 129, 139, 150, 160, 171, 181, 192, 193, 203, 224, 246, 256):
    a = cupy.random.rand(n, g)
    b = cupy.random.rand(n, g)
    out = cupy.empty((n, n))
    for _ in range(3):
        a.dot(b.T, out=out)
    ts = []
    for _ in range(args.repeat):
        e0, e1 = cupy.cuda.Event(), cupy.cuda.Event()
        e0.record()
        a.dot(b.T, out=out)
        e1.record()
        e1.synchronize()
        ts.append(cupy.cuda.get_elapsed_time(e0, e1) * 1e-3)
    t = sorted(ts)[len(ts) // 2]
    m_pad, n_pad = -(-n // 128) * 128, -(-n // 64) * 64
    print(f'{n:4d} {t * 1e3:7.3f} {2 * n * n * g / t / 1e9:14.1f} {2 * m_pad * n_pad * g / t / 1e9:30.1f}')
