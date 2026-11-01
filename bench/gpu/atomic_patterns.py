"""Throughput of FP64 atomic adds and loads by access pattern (node1), for the CPHF J/K's density loop: every
thread adds 1.0 (or loads) at addresses from a hash, over an array of 2^LOG2N doubles, in the patterns
  scattered   each lane its own random address (the J+K kernels' K and kl-indexed J atomics)
  warp-same   one random address per warp instruction, shared by its 32 lanes (the ij-indexed J atomics)
  coalesced   32 consecutive doubles per warp instruction, at a random 256-byte block
  thread-seq  each lane a random base and consecutive addresses over NSEQ iterations (a density-fastest layout
              with the density loop inside the thread)
Prints G thread operations per second per pattern, for the array in DRAM (2^24 doubles, 128 MB) and in L2 (2^16,
512 KB), at full occupancy and at 8 warps per SM (the kernels' occupancy).
Usage: atomic_patterns.py [--niter 1024] [--nseq 72] [--repeat 5]"""
import argparse
import cupy

ap = argparse.ArgumentParser()
ap.add_argument('--niter', type=int, default=1024)
ap.add_argument('--nseq', type=int, default=72)
ap.add_argument('--repeat', type=int, default=5)
args = ap.parse_args()

src = r'''
__device__ __forceinline__ unsigned mix(unsigned h)
{
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}
__device__ __forceinline__ unsigned long long address(int pattern, unsigned tid, int it, int nseq,
                                                      unsigned long long mask)
{
    unsigned lane = tid & 31;
    unsigned warp = tid >> 5;
    switch (pattern) {
    case 0: return mix(tid * 0x9e3779b9u + it) & mask;
    case 1: return mix(warp * 0x9e3779b9u + it) & mask;
    case 2: return ((mix(warp * 0x9e3779b9u + it) << 5) + lane) & mask;
    default: return ((unsigned long long)(mix(tid * 0x9e3779b9u + it / nseq)) * nseq + it % nseq) & mask;
    }
}
extern "C" __global__
void atom_patterns(double *v, unsigned long long mask, int pattern, int niter, int nseq)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = 0; it < niter; ++it) {
        atomicAdd(v + address(pattern, tid, it, nseq, mask), 1.);
    }
}
extern "C" __global__
void load_patterns(const double *v, double *out, unsigned long long mask, int pattern, int niter, int nseq)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    double acc = 0;
    for (int it = 0; it < niter; ++it) {
        acc += v[address(pattern, tid, it, nseq, mask)];
    }
    out[tid] = acc;
}
'''
mod = cupy.RawModule(code=src)
atom = mod.get_function('atom_patterns')
load = mod.get_function('load_patterns')
dev = cupy.cuda.Device()
nsm = dev.attributes['MultiProcessorCount']
patterns = ['scattered', 'warp-same', 'coalesced', 'thread-seq']
print(f'{nsm} SMs, {args.niter} operations per thread, thread-seq runs of {args.nseq}')
print(f"{'array':>10s} {'warps/SM':>8s} {'op':>5s} " + ' '.join(f'{p:>11s}' for p in patterns) + '   (G ops/s)')
for log2n in (24, 16):
    v = cupy.zeros(1 << log2n)
    for warps_per_sm, block in ((32, 256), (8, 256)):
        nblk = nsm * warps_per_sm * 32 // block
        out = cupy.zeros(nblk * block)
        for op, fn in (('atom', atom), ('load', load)):
            rates = []
            for pattern in range(4):
                def run():
                    if op == 'atom':
                        fn((nblk,), (block,), (v, cupy.uint64((1 << log2n) - 1), cupy.int32(pattern),
                                               cupy.int32(args.niter), cupy.int32(args.nseq)))
                    else:
                        fn((nblk,), (block,), (v, out, cupy.uint64((1 << log2n) - 1), cupy.int32(pattern),
                                               cupy.int32(args.niter), cupy.int32(args.nseq)))
                run()
                times = []
                for _ in range(args.repeat):
                    e0, e1 = cupy.cuda.Event(), cupy.cuda.Event()
                    e0.record()
                    run()
                    e1.record()
                    e1.synchronize()
                    times.append(cupy.cuda.get_elapsed_time(e0, e1) * 1e-3)
                rates.append(nblk * block * args.niter / sorted(times)[len(times) // 2] / 1e9)
            size = f'{8 << log2n >> 20} MB' if log2n >= 17 else f'{8 << log2n >> 10} KB'
            print(f'{size:>10s} {warps_per_sm:8d} {op:>5s} ' + ' '.join(f'{r:11.2f}' for r in rates))
