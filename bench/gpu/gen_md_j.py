"""Generator of MD J-engine kernels for GPU4PySCF (gpu4pyscf/lib/gvhf-md/generated_md_j.cu).

GPU4PySCF's unrolled MD J kernels (unrolled_md_j.cu) cover the classes with lij + lkl <= 5; the others (the f-function
classes of def2-TZVP, and (dp|dp), (dd|ds), ...) run the generic md_j_1dm_kernel, which takes its R-tensor and
contraction addresses from index tables at run time (part 23's profile: FP64 is 10% of its instructions, the FP64 pipe
61-66% busy, against 88-90% in the unrolled kernels). This script writes kernels for any (lij, lkl):
  - the launch shape, tiles, screening and reductions of the unrolled kernels: 16 x 16 threads, one (bra pair, ket pair)
    quartet per thread, bra tiles of run-time length (part 22) and TILEY batches of 16 ket pairs per block;
  - the Boys function into a local array (registers), not shared memory;
  - the R tensor by the McMurchie-Davidson recursions, z first, then y, then x, column by column (u, v), so that only a
    few levels of intermediates are live; each R^(0)_tuv is contracted at once into both sides' accumulators,
        vj_kl[k] += R[i+k] dm_ij[i]   (the bra density from shared memory, reduced over the 16 bra lanes per quartet)
        vj_ij[i] += R[i+k] dm_kl[k]   (the ket density in registers, the phase (-1)^|k| folded in)
    where the unrolled kernels keep the whole R tensor live until the end of the quartet.
A dispatcher md_j_generated() launches them; MD_build_j calls it after md_j_unrolled() and before the generic kernel
(single density only; several densities keep md_j_4dm_kernel).
A spec gLIJ,LKL writes the gradient's kernel md_j_ip1_LIJ_LKL instead (r_tensor's ip1, dispatched by
md_j_generated_ip1, which MD_build_j_ip1 tries first); one file holds one kind.
--prefactor=old keeps the unrolled kernels' quartet prefactor (two divisions and a sqrt) instead of part 27's.
Usage: gen_md_j.py OUT.cu [--prefactor=rsqrt|old] [g]LIJ,LKL[:TILEY] ...   e.g. gen_md_j.py generated_md_j.cu 3,3 4,2:16"""
import sys

# the largest bra tile in batches of 16 pairs (the run-time tile is cut to fill the SMs, part 22)
TILEX = 48
# rsqrt: each quartet's prefactor and theta from 1/aij, 1/akl kept with the pair data and one rsqrt (part 27), the
# caches one row longer on each side; old: fac / (aij*akl*sqrt(aij+akl)) and aij*akl/(aij+akl) as the unrolled kernels had
PREFACTOR = 'rsqrt'


def nf3(l):
    return (l + 1) * (l + 2) * (l + 3) // 6


def herm(l):
    '''Hermite components (t, u, v) with t + u + v <= l, in GPU4PySCF's order (t outermost, v innermost)'''
    return [(t, u, v) for t in range(l + 1) for u in range(l + 1 - t) for v in range(l + 1 - t - u)]


def r_tensor(lij, lkl, ip1=False):
    '''The R recursions with the contraction of each R^(0) as soon as it is computed: lines of code. With ip1 (the
    J engine's gradient) both potentials reach one order more than the densities, R one order more than the J build's:
    the bra potential (order lij+1) from the ket density (order lkl), the ket potential (order lkl+1) from the bra density
    (order lij), all in the layout of one more order per pair.'''
    L = lij + lkl + (1 if ip1 else 0)
    bra = {c: n for n, c in enumerate(herm(lij + (1 if ip1 else 0)))}
    ket = list(enumerate(herm(lkl)))
    ket_out = {c: n for n, c in enumerate(herm(lkl + 1))}
    out = []

    def name(n, t, u, v):
        return f'gamma_inc[{n}]' if t == u == v == 0 else f'r{n}_{t}_{u}_{v}'

    def emit(n, t, u, v):
        '''R^(n)_tuv from the recursion along the last nonzero of z (v), y (u), x (t)'''
        if t > 0:
            d, a, b, c = 'xpq', (n + 1, t - 1, u, v), (n + 1, t - 2, u, v), t - 1
        elif u > 0:
            d, a, b, c = 'ypq', (n + 1, t, u - 1, v), (n + 1, t, u - 2, v), u - 1
        else:
            d, a, b, c = 'zpq', (n + 1, t, u, v - 1), (n + 1, t, u, v - 2), v - 1
        expr = f'{d} * {name(*a)}'
        if c == 1:
            expr += f' + {name(*b)}'
        elif c > 1:
            expr += f' + {c} * {name(*b)}'
        out.append(f'            double {name(n, t, u, v)} = {expr};')

    def consume(t, u, v):
        r = name(0, t, u, v)
        if ip1:
            for kk, (kt, ku, kv) in ket:      # the bra potential from the ket density
                i = (t - kt, u - ku, v - kv)
                if min(i) >= 0 and sum(i) <= lij + 1:
                    out.append(f'            vj_ij[{bra[i]}] += {r} * dkl[{kk}];')
            for i in herm(lij):               # the ket potential from the bra density
                k = (t - i[0], u - i[1], v - i[2])
                if min(k) >= 0 and sum(k) <= lkl + 1:
                    out.append(f'            vkl[{ket_out[k]}] += {r} * dm_ij_cache[{bra[i] * 16}];')
            return
        for kk, (kt, ku, kv) in ket:
            i = (t - kt, u - ku, v - kv)
            if min(i) < 0 or sum(i) > lij:
                continue
            ii = bra[i]
            out.append(f'            vkl[{kk}] += {r} * dm_ij_cache[{ii * 16}];')
            out.append(f'            vj_ij[{ii}] += {r} * dkl[{kk}];')

    for v in range(L + 1):
        if v > 0:
            for n in range(L - v + 1):
                emit(n, 0, 0, v)
        for u in range(L - v + 1):
            if u > 0:
                for n in range(L - v - u + 1):
                    emit(n, 0, u, v)
            consume(0, u, v)
            for t in range(1, L - v - u + 1):
                for n in range(L - v - u - t + 1):
                    emit(n, t, u, v)
                consume(t, u, v)
    return out


def kernel(lij, lkl, tiley, ip1=False):
    up = 1 if ip1 else 0
    L = lij + lkl + up
    nij, nkl = nf3(lij + up), nf3(lkl + up)     # the potentials' (and with ip1 the densities' layout) components
    bsy = 16 * tiley
    diag = lij == lkl
    ket = herm(lkl)
    ket_pos = [herm(lkl + up).index(c) for c in ket]   # the ket density's components in its layout
    name = f'md_j_ip1_{lij}_{lkl}' if ip1 else f'md_j_gen_{lij}_{lkl}'
    rs = PREFACTOR == 'rsqrt'
    rq_rows, rp_len = (5, 80) if rs else (4, 64)
    s = []
    s.append(f'// lij={lij}, lkl={lkl}{" (gradient: the potentials one order up)" if ip1 else ""}: '
             f'TILEX={TILEX} (largest; run-time tilex), TILEY={tiley}\n'
             '__global__ __launch_bounds__(256, 1) static\n'
             f'void {name}(KERNEL_ARGS)\n{{\n'
             '    KERNEL_SETUP();\n'
             '    // the bra tile, from the grid that md_j_gen_grid_x chose\n'
             '    int tilex = ((bounds.npairs_ij + 15) / 16 + gridDim.x - 1) / gridDim.x;\n'
             '    int *pair_ij_mapping = bounds.pair_ij_mapping;\n'
             '    int *pair_kl_mapping = bounds.pair_kl_mapping;\n'
             '    int task_ij0 = block_x * (16*tilex);\n'
             f'    int task_kl0 = block_y * {bsy};\n'
             '    if (q_cond_ij[task_ij0] + q_cond_kl[task_kl0] < bounds.cutoff) {\n'
             '        return;\n'
             '    }\n')
    if diag:
        s.append('    if (pair_ij_mapping == pair_kl_mapping && task_ij0+16*tilex <= task_kl0) {\n'
                 '        return;\n'
                 '    }\n')
    s.append('    int sq_id = tx + 16 * ty;\n'
             '    int thread_id = sq_id;\n'
             '    int *bas = envs.bas;\n'
             '    int *pair_ij_loc = bounds.pair_ij_loc;\n'
             '    int *pair_kl_loc = bounds.pair_kl_loc;\n'
             '    int nbas = envs.nbas;\n'
             '    double *env = envs.env;\n'
             '    double *vj = jk.vj;\n'
             '    unsigned int lane_id = thread_id % 32;\n'
             '    unsigned int group_id = lane_id / 16;\n'
             '    unsigned int mask = 0xffff << (group_id * 16);\n'
             '\n'
             '    int npairs_ij = bounds.npairs_ij;\n'
             '    int npairs_kl = bounds.npairs_kl;\n'
             f'    double *Rq_cache = vj_kl_cache + {nkl * bsy};\n'
             f'    double *Rp_cache = Rq_cache + {rq_rows * bsy};\n'
             f'    double *dm_ij_cache = Rp_cache + {rp_len} + tx;\n'
             '    float *qd_ij_max = bounds.qd_ij_max;\n'
             '    float *qd_kl_max = bounds.qd_kl_max;\n'
             f'    for (int n = thread_id; n < {(nkl + rq_rows) * bsy + rp_len}; n += 256) {{\n'
             '        vj_kl_cache[n] = 0.;\n'
             '    }\n'
             '    __syncthreads();\n'
             '\n'
             f'    for (int n = thread_id; n < {bsy}; n += 256) {{\n'
             f'        int task_kl = blockIdx.y * {bsy} + n;\n'
             '        if (task_kl < npairs_kl) {\n'
             '            int pair_kl = pair_kl_mapping[task_kl];\n'
             '            int ksh = pair_kl / nbas;\n'
             '            int lsh = pair_kl % nbas;\n'
             '            double ak = env[bas[ksh*BAS_SLOTS+PTR_EXP]];\n'
             '            double al = env[bas[lsh*BAS_SLOTS+PTR_EXP]];\n'
             '            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];\n'
             '            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];\n'
             '            double akl = ak + al;\n'
             '            double xkl = (ak * rk[0] + al * rl[0]) / akl;\n'
             '            double ykl = (ak * rk[1] + al * rl[1]) / akl;\n'
             '            double zkl = (ak * rk[2] + al * rl[2]) / akl;\n'
             '            Rq_cache[n+0] = xkl;\n'
             f'            Rq_cache[n+{bsy}] = ykl;\n'
             f'            Rq_cache[n+{2 * bsy}] = zkl;\n'
             f'            Rq_cache[n+{3 * bsy}] = akl;\n'
             + (f'            Rq_cache[n+{4 * bsy}] = 1. / akl;\n' if rs else '') +
             '        } else {\n'
             '            Rq_cache[n+0] = 2e5;\n'
             f'            Rq_cache[n+{bsy}] = 2e5;\n'
             f'            Rq_cache[n+{2 * bsy}] = 2e5;\n'
             f'            Rq_cache[n+{3 * bsy}] = 1.;\n'
             + (f'            Rq_cache[n+{4 * bsy}] = 1.;\n' if rs else '') +
             '        }\n'
             '    }\n'
             '\n'
             '    for (int batch_ij = 0; batch_ij < tilex; ++batch_ij) {\n'
             '        int task_ij0 = blockIdx.x * (16*tilex) + batch_ij * 16;\n'
             '        if (task_ij0 >= npairs_ij) {\n'
             '            break;\n'
             '        }\n'
             '        __syncthreads();\n'
             '        if (thread_id < 16) {\n'
             '            int task_ij = task_ij0 + thread_id;\n'
             '            if (task_ij < npairs_ij) {\n'
             '                int pair_ij = pair_ij_mapping[task_ij];\n'
             '                int ish = pair_ij / nbas;\n'
             '                int jsh = pair_ij % nbas;\n'
             '                double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];\n'
             '                double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];\n'
             '                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];\n'
             '                double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];\n'
             '                double aij = ai + aj;\n'
             '                double xij = (ai * ri[0] + aj * rj[0]) / aij;\n'
             '                double yij = (ai * ri[1] + aj * rj[1]) / aij;\n'
             '                double zij = (ai * ri[2] + aj * rj[2]) / aij;\n'
             '                Rp_cache[thread_id+0] = xij;\n'
             '                Rp_cache[thread_id+16] = yij;\n'
             '                Rp_cache[thread_id+32] = zij;\n'
             '                Rp_cache[thread_id+48] = aij;\n'
             + ('                Rp_cache[thread_id+64] = 1. / aij;\n' if rs else '') +
             '            } else {\n'
             '                Rp_cache[thread_id+0] = 2e5;\n'
             '                Rp_cache[thread_id+16] = 2e5;\n'
             '                Rp_cache[thread_id+32] = 2e5;\n'
             '                Rp_cache[thread_id+48] = 1.; // aij\n'
             + ('                Rp_cache[thread_id+64] = 1.;\n' if rs else '') +
             '            }\n'
             '        }\n'
             '        int task_ij = task_ij0 + tx;\n'
             '        if (task_ij >= npairs_ij) {\n'
             '            task_ij = task_ij0;\n'
             '        }\n'
             '        int ij_loc0 = pair_ij_loc[task_ij];\n'
             '        double *dm = jk.dm + ij_loc0;\n'
             f'        for (int n = ty; n < {nij}; n += 16) {{\n'
             '            dm_ij_cache[n*16] = dm[n];\n'
             '        }\n'
             f'        double vj_ij[{nij}];\n'
             '#pragma unroll\n'
             f'        for (int ij = 0; ij < {nij}; ++ij) {{\n'
             '            vj_ij[ij] = 0;\n'
             '        }\n'
             f'        for (int batch_kl = 0; batch_kl < {tiley}; ++batch_kl) {{\n'
             f'            int task_kl0 = blockIdx.y * {bsy} + batch_kl * 16;\n'
             '            if (task_kl0 >= npairs_kl) {\n'
             '                break;\n'
             '            }\n')
    if diag:
        s.append('            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {\n'
                 '                break;\n'
                 '            }\n')
    s.append('            if (qd_ij_max[batch_ij+blockIdx.x*tilex] + q_cond_kl[task_kl0] < bounds.cutoff &&\n'
             f'                qd_kl_max[batch_kl+blockIdx.y*{tiley}] + q_cond_ij[task_ij0] < bounds.cutoff) {{\n'
             '                continue;\n'
             '            }\n'
             '\n'
             '            int sq_kl = ty + batch_kl * 16;\n'
             '            int task_ij = task_ij0 + tx;\n'
             '            int task_kl = task_kl0 + ty;\n'
             '            double fac = PI_FAC;\n'
             '            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {\n'
             '                fac = 0.;\n'
             '            }\n')
    if diag:
        s.append('            if (pair_ij_mapping == pair_kl_mapping) {\n'
                 '                if (task_ij == task_kl) fac *= .5;\n'
                 '                else if (task_ij < task_kl) fac = 0.;\n'
                 '            }\n')
    s.append('            __syncthreads();\n'
             '            double xij = Rp_cache[tx+0];\n'
             '            double yij = Rp_cache[tx+16];\n'
             '            double zij = Rp_cache[tx+32];\n'
             '            double aij = Rp_cache[tx+48];\n'
             + ('            double inv_aij = Rp_cache[tx+64];\n' if rs else '') +
             '            double xkl = Rq_cache[sq_kl+0];\n'
             f'            double ykl = Rq_cache[sq_kl+{bsy}];\n'
             f'            double zkl = Rq_cache[sq_kl+{2 * bsy}];\n'
             f'            double akl = Rq_cache[sq_kl+{3 * bsy}];\n'
             + (f'            double inv_akl = Rq_cache[sq_kl+{4 * bsy}];\n'
                '            double rs = rsqrt(aij+akl);\n'
                '            fac = fac * (inv_aij * inv_akl * rs);\n' if rs else
                '            fac = fac / (aij*akl*sqrt(aij+akl));\n') +
             '            double xpq = xij - xkl;\n'
             '            double ypq = yij - ykl;\n'
             '            double zpq = zij - zkl;\n'
             '            double rr = xpq*xpq + ypq*ypq + zpq*zpq;\n'
             + ('            double theta = aij * akl * (rs * rs);\n' if rs else
                '            double theta = aij * akl / (aij + akl);\n') +
             '            double omega = jk.omega;\n'
             f'            double gamma_inc[{2 * (L + 1)}];\n'
             f'            boys_fn(gamma_inc, theta, rr, omega, fac, {L}, 0, 1);\n'
             '            // the ket pair\'s Hermite density, with the phase (-1)^|k| of its components\n'
             '            const double *dm_kl = jk.dm + pair_kl_loc[task_kl < npairs_kl ? task_kl : task_kl0];\n'
             f'            double dkl[{len(ket)}];\n')
    for kk, c in enumerate(ket):
        sign = '-' if sum(c) % 2 else ''
        s.append(f'            dkl[{kk}] = {sign}dm_kl[{ket_pos[kk]}];\n')
    s.append(f'            double vkl[{nkl}];\n'
             '#pragma unroll\n'
             f'            for (int kl = 0; kl < {nkl}; ++kl) {{\n'
             '                vkl[kl] = 0.;\n'
             '            }\n')
    s.append('\n'.join(r_tensor(lij, lkl, ip1)) + '\n')
    for kk, c in enumerate(herm(lkl + up)):
        op = '-=' if sum(c) % 2 else '+='
        s.append('            for (int offset = 8; offset > 0; offset /= 2) {\n'
                 f'                vkl[{kk}] += __shfl_down_sync(mask, vkl[{kk}], offset);\n'
                 '            }\n'
                 f'            if (tx == 0) {{ vj_kl_cache[sq_kl+{kk * bsy}] {op} vkl[{kk}]; }}\n')
    s.append('        }\n'
             '        {\n'
             '        double *vj_cache = Rp_cache;\n'
             '        int task_ij = task_ij0 + tx;\n'
             '#pragma unroll\n'
             f'        for (int n = 0; n < {nij}; ++n) {{\n'
             '            __syncthreads();\n'
             '            vj_cache[thread_id] = vj_ij[n];\n'
             '            for (int stride = 8; stride > 0; stride /= 2) {\n'
             '                __syncthreads();\n'
             '                if (ty < stride) {\n'
             '                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];\n'
             '                }\n'
             '            }\n'
             '            __syncthreads();\n'
             '            if (ty == 0 && task_ij < npairs_ij) {\n'
             '                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);\n'
             '            }\n'
             '        }\n'
             '        }\n'
             '    }\n'
             '    __syncthreads();\n'
             f'    for (int n = tx; n < {nkl * tiley}; n += 16) {{\n'
             f'        int kl = n / {tiley};\n'
             f'        int batch_kl = n - kl * {tiley};\n'
             '        int sq_kl = ty + batch_kl * 16;\n'
             f'        int task_kl = blockIdx.y * {bsy} + sq_kl;\n'
             '        if (task_kl < npairs_kl) {\n'
             '            int kl_loc0 = pair_kl_loc[task_kl];\n'
             f'            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*{bsy}]);\n'
             '        }\n'
             '    }\n'
             '}\n')
    return ''.join(s)


HEAD = '''// Generated by bench/gpu/gen_md_j.py (my-pyscf workspace): MD J kernels for classes without an unrolled kernel.
#include <stdio.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-md/boys.cu"
#include "gvhf-md/md_j.cuh"

#define KERNEL_ARGS \\
    RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, \\
    float *q_cond_ij, float *q_cond_kl

#define KERNEL_SETUP() \\
    int tx = threadIdx.x; \\
    int ty = threadIdx.y; \\
    int block_x = blockIdx.x; \\
    int block_y = blockIdx.y; \\
    extern __shared__ double vj_kl_cache[];

'''

GRID_X = '''// gridDim.x of a generated J launch: bra tiles of at most tilex_max batches of 16 pairs, enough of them that the
// grid has at least MD_J_GEN_BLOCKS_PER_SM blocks per SM (twice that if bra and ket share their pairs), and only as
// many as the tiles of ceil(nbatch / gridDim.x) batches fill (as md_j_grid_x in unrolled_md_j.cu)
#define MD_J_GEN_BLOCKS_PER_SM 64
static int md_j_gen_grid_x(int npairs_ij, int tilex_max, int grid_y, int diagonal)
{
    int dev, n_sm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
    int nbatch = (npairs_ij + 15) / 16;
    int gx = (nbatch + tilex_max - 1) / tilex_max;
    int want = MD_J_GEN_BLOCKS_PER_SM * n_sm * (diagonal ? 2 : 1);
    if (gx * grid_y < want && nbatch > 0) {
        gx = std::min(nbatch, (want + grid_y - 1) / grid_y);
        int tilex = (nbatch + gx - 1) / gx;
        gx = (nbatch + tilex - 1) / tilex;
    }
    return std::max(gx, 1);
}

'''


def default_tiley(lkl):
    '''About 40 KB of shared memory for the ket tile (vj_kl cache and pair data), at most 32 batches; lkl is the
    order of the ket's potential'''
    return max(1, min(32, 5000 // (16 * (nf3(lkl) + 4))))


def main():
    global PREFACTOR
    out = sys.argv[1]
    specs = []
    for a in sys.argv[2:]:
        if a.startswith('--prefactor='):
            PREFACTOR = a.split('=', 1)[1]
            assert PREFACTOR in ('rsqrt', 'old')
            continue
        cls, _, ty = a.partition(':')
        ip1 = cls.startswith('g')
        lij, lkl = map(int, cls.lstrip('g').split(','))
        specs.append((lij, lkl, int(ty) if ty else default_tiley(lkl + ip1), ip1))
    modes = {sp[3] for sp in specs}
    assert len(modes) == 1, 'J and gradient kernels go to separate files'
    ip1 = modes.pop()
    src = [HEAD]
    for lij, lkl, ty, _ in specs:
        src.append(kernel(lij, lkl, ty, ip1))
        src.append('\n')
    src.append(GRID_X)
    if ip1:
        # MD_build_j_ip1 has raised the pairs' orders by one before this is called
        src.append('int md_j_generated_ip1(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds,\n'
                   '                       float *q_cond_ij, float *q_cond_kl, double omega)\n{\n')
        key = '(bounds->lij-1)*11 + bounds->lkl-1'
    else:
        src.append('int md_j_generated(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds,\n'
                   '                   float *q_cond_ij, float *q_cond_kl, double omega)\n{\n')
        key = 'bounds->lij*11 + bounds->lkl'
    src.append('    int npairs_ij = bounds->npairs_ij;\n'
               '    int npairs_kl = bounds->npairs_kl;\n'
               '    int diagonal = bounds->pair_ij_mapping == bounds->pair_kl_mapping;\n'
               '    dim3 threads(16, 16);\n'
               f'    switch ({key}) {{\n')
    for lij, lkl, ty, _ in specs:
        bsy = 16 * ty
        name = f'md_j_ip1_{lij}_{lkl}' if ip1 else f'md_j_gen_{lij}_{lkl}'
        # the Rp region also holds the bra-side reduction's 256 doubles (vj_cache = Rp_cache)
        rq_rows, rp_len = (5, 80) if PREFACTOR == 'rsqrt' else (4, 64)
        shm = (nf3(lkl + ip1) + rq_rows) * bsy + max(rp_len + 16 * nf3(lij + ip1), 256)
        src.append(f'    case {lij * 11 + lkl}: {{ // TILEY={ty}, {shm * 8} bytes of shared memory\n'
                   f'        dim3 blocks(md_j_gen_grid_x(npairs_ij, {TILEX}, (npairs_kl + {bsy - 1}) / {bsy}, diagonal),\n'
                   f'                    (npairs_kl + {bsy - 1}) / {bsy}, 1);\n'
                   f'        cudaFuncSetAttribute({name}, cudaFuncAttributeMaxDynamicSharedMemorySize, {shm * 8});\n'
                   f'        {name}<<<blocks, threads, {shm * 8}>>>(*envs, *jk, *bounds, q_cond_ij, q_cond_kl);\n'
                   '    } break;\n')
    src.append('    default: return 0;\n'
               '    }\n'
               '    return 1;\n'
               '}\n')
    open(out, 'w').write(''.join(src))
    for lij, lkl, ty, _ in specs:
        print(f'{"md_j_ip1" if ip1 else "md_j_gen"}_{lij}_{lkl}: TILEY {ty}, {nf3(lij)} x {nf3(lkl)} Hermite components, '
              f'order {lij + lkl + ip1}')


if __name__ == '__main__':
    main()
