"""Generator of multi-thread Rys K kernels for GPU4PySCF (gpu4pyscf/lib/gvhf-rys/generated_rys_k.cu).

The unrolled multi-thread kernels of GPU4PySCF (rys_k_2021, ..., rys_k_3110 in unrolled_rys_k.cu) cover six
classes; every other class with more outputs than one thread can hold goes to the generic rys_k_kernel, which
computes its addresses from index arrays at run time (FP64 is 16-17% of its instructions, against 39-45% in the
unrolled kernels). This script writes kernels in the style of the unrolled ones for any (li, lj, lk, ll):
  - nsq quartets per block, gs threads per quartet (blockDim (nsq, gs)); nsq is 32 or 64 so that a warp holds one
    gout_id and the switch over gout_id below never diverges;
  - the shared-memory layout, the primitive loops and the bra-pair skip of the unrolled kernels;
  - the Rys recursions for one Cartesian direction per thread (threads 0-2), fully unrolled with constant offsets,
    in the in-place layout of the generic kernel: g(i, k) at i + k*stride_k after the TRR, then the HRR to (i, j) and
    (k, l) (stride_j = li+1, stride_k = stride_j*(lj+1), stride_l = stride_k*(lk+1));
  - the outputs (i fastest, lexical Cartesian order) dealt round robin to the gs threads, each accumulating its own
    with constant addresses;
  - the K contraction of the generic kernel (vk_il from dm_jk, vk_ik from dm_jl, and when ish != jsh vk_jl from dm_ik
    and vk_jk from dm_il), unrolled per thread with one atomicAdd per element.
A dispatcher rys_k_generated() launches them; RYS_build_k calls it after rys_k_unrolled() and before the generic
kernel. GPU4PYSCF_RYS_GEN_FIRST=1 (experiment builds) lets it run before rys_k_unrolled, to compare against the
unrolled kernels of the same classes.
Usage: gen_rys_k.py OUT.cu CLASS[:NSQxGS] ...   e.g. gen_rys_k.py generated_rys_k.cu 3100:64x4 3111:32x8"""
import math
import sys


def cart(l):
    '''Cartesian components of angular momentum l in lexical order: (lx, ly, lz)'''
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def recursions(li, lj, lk, ll, nsq):
    '''The TRR and HRR of one Cartesian direction, writing into _gx (the direction's g array)'''
    lij, lkl = li + lj, lk + ll
    sj = li + 1
    sk = sj * (lj + 1)
    sl = sk * (lk + 1)

    def g(idx):
        return f'_gx[{idx * nsq}]'
    out = []
    if lij > 0:
        out += [f's0 = {g(0)};', 's1 = c0x * s0;', f'{g(1)} = s1;']
        for i in range(1, lij):
            out += [f's2 = c0x * s1 + {i} * b10 * s0;', f'{g(i + 1)} = s2;']
            if i < lij - 1:
                out += ['s0 = s1;', 's1 = s2;']
    if lkl > 0:
        out += [f'double xlxk = rlrk[n*{nsq}];', 'double Rqc = xlxk * al_akl;',
                f'double cpx = Rqc + rt_akl * Rpq[n*{nsq}];']
        for i in range(lij + 1):
            out += [f's0 = {g(i)};', 's1 = cpx * s0;']
            if i > 0:
                out += [f's1 += {i} * b00 * {g(i - 1)};']
            out += [f'{g(i + sk)} = s1;']
            for k in range(1, lkl):
                out += [f's2 = cpx*s1 + {k} * b01 *s0;']
                if i > 0:
                    out += [f's2 += {i} * b00 * {g(i - 1 + k * sk)};']
                out += [f'{g(i + (k + 1) * sk)} = s2;']
                if k < lkl - 1:
                    out += ['s0 = s1;', 's1 = s2;']
    # HRR i -> j at every k level of the TRR
    for k in range(lkl + 1):
        for j in range(lj):
            ij = lij + j * li
            out += [f's1 = {g(k * sk + ij)};']
            for ij2 in range(ij - 1, j * sj - 1, -1):
                out += [f's0 = {g(k * sk + ij2)};', f'{g(k * sk + ij2 + sj)} = s1 - xjxi * s0;', 's1 = s0;']
    # HRR k -> l for every (i, j)
    for i in range(sk):
        for l in range(ll):
            kl = (lkl + l * lk) * sk
            out += [f's1 = {g(i + kl)};']
            for kl2 in range(kl - sk, l * sl - 1, -sk):
                out += [f's0 = {g(i + kl2)};', f'{g(i + kl2 + sl)} = s1 - xlxk * s0;', 's1 = s0;']
    return out


def kernel(li, lj, lk, ll, nsq, gs):
    name = f'rys_k_gen_{li}{lj}{lk}{ll}'
    lij, lkl = li + lj, lk + ll
    sj = li + 1
    sk = sj * (lj + 1)
    sl = sk * (lk + 1)
    G = sl * (ll + 1)
    ci, cj, ck, cl = cart(li), cart(lj), cart(lk), cart(ll)
    outs = [(i, j, k, l) for l in range(len(cl)) for k in range(len(ck)) for j in range(len(cj)) for i in range(len(ci))]
    per = -(-len(outs) // gs)
    mine = [[(m // gs, outs[m]) for m in range(t, len(outs), gs)] for t in range(gs)]

    def addr(o, d):
        i, j, k, l = o
        return ci[i][d] + cj[j][d] * sj + ck[k][d] * sk + cl[l][d] * sl
    I1, I2, I3, I4 = ' ' * 4, ' ' * 8, ' ' * 12, ' ' * 16
    L = [f'// ({"spdfghi"[li]}{"spdfghi"[lj]}|{"spdfghi"[lk]}{"spdfghi"[ll]}): {len(outs)} outputs, {gs} threads x {nsq} '
         f'quartets per block, {per} outputs per thread, g_size {G}',
         '__global__ static', f'void {name}(JKMATRIX_KERNEL_ARGS)', '{', '    JKMATRIX_KERNEL_SETUP();', '',
         f'    int t_id = {nsq} * gout_id + sq_id;', f'    constexpr int threads = {nsq * gs};',
         f'    constexpr int nsq_per_block = {nsq};', f'    constexpr int g_size = {G};',
         '    uint32_t nbas = envs.nbas;', '    int *bas = envs.bas;', '    double *env = envs.env;', '',
         '    double *rlrk = shared_memory + sq_id;', '    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;',
         '    double *gx = shared_memory + nsq_per_block * 6 + sq_id;',
         '    double *rw = shared_memory + nsq_per_block * (g_size*3+6) + sq_id;',
         '    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+bounds.nroots*2+6);']
    L += PROLOGUE.splitlines()
    L += [f'        double gout[{per}];', '#pragma unroll', f'        for (int n = 0; n < {per}; ++n) {{ gout[n] = 0; }}']
    L += PRIM_LOOPS.replace('ROOTS_STRIDE', str(gs)).splitlines()
    # root loop
    body = [f'{I4}for (int irys = 0; irys < nroots; ++irys) {{', f'{I4}    __syncthreads();', f'{I4}    double s0, s1, s2;',
            f'{I4}    double rt = rw[irys*{2 * nsq}];', f'{I4}    double rt_aa = rt / (aij + akl);',
            f'{I4}    double rt_aij = rt_aa * akl;']
    if lij > 1:
        body += [f'{I4}    double b10 = .5/aij * (1 - rt_aij);']
    if lkl > 0:
        body += [f'{I4}    double rt_akl = rt_aa * aij;']
        if lij > 0:
            body += [f'{I4}    double b00 = .5 * rt_aa;']
        if lkl > 1:
            body += [f'{I4}    double b01 = .5/akl * (1 - rt_akl);']
    body += [f'{I4}    for (int n = gout_id; n < 3; n += {gs}) {{', f'{I4}        if (n == 2) {{',
             f'{I4}            gx[{2 * G * nsq}] = rw[irys*{2 * nsq}+{nsq}];', f'{I4}        }}',
             f'{I4}        double *_gx = gx + n * {G * nsq};', f'{I4}        double xjxi = rjri[n];',
             f'{I4}        double Rpa = xjxi * aj_aij;', f'{I4}        double c0x = Rpa - rt_aij * Rpq[n*{nsq}];']
    body += [f'{I4}        {s}' for s in recursions(li, lj, lk, ll, nsq)]
    body += [f'{I4}    }}', f'{I4}    __syncthreads();', f'{I4}    switch (gout_id) {{']
    for t in range(gs):
        body += [f'{I4}    case {t}:']
        for slot, o in mine[t]:
            body += [f'{I4}    gout[{slot}] += gx[{addr(o, 0) * nsq}] * gx[{(G + addr(o, 1)) * nsq}] * '
                     f'gx[{(2 * G + addr(o, 2)) * nsq}];']
        body += [f'{I4}    break;']
    body += [f'{I4}    }}', f'{I4}}}']
    L += body + [f'{I3}}}', f'{I2}}}']
    # K contraction
    L += [f'{I2}if (task_id < ntasks) {{', f'{I3}int *ao_loc = envs.ao_loc;', f'{I3}int nao = ao_loc[nbas];',
          f'{I3}int i0 = ao_loc[ish];', f'{I3}int j0 = ao_loc[jsh];', f'{I3}int k0 = ao_loc[ksh];', f'{I3}int l0 = ao_loc[lsh];',
          f'{I3}size_t nao2 = (size_t)nao * nao;', f'{I3}for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {{',
          f'{I4}double *dm = jk.dm + i_dm * nao2;', f'{I4}double *vk = jk.vk + i_dm * nao2;']
    # (target pair, dm pair) per contraction: indices into (i, j, k, l)
    contractions = [((0, 3), (1, 2), 'i0', 'l0', 'j0', 'k0'), ((0, 2), (1, 3), 'i0', 'k0', 'j0', 'l0'),
                    ((1, 3), (0, 2), 'j0', 'l0', 'i0', 'k0'), ((1, 2), (0, 3), 'j0', 'k0', 'i0', 'l0')]
    for c, (tgt, dmp, ta, tb, da, db) in enumerate(contractions):
        if c == 2:
            L += [f'{I4}if (ish != jsh) {{']
        ind = I4 + ('    ' if c >= 2 else '')
        L += [f'{ind}switch (gout_id) {{']
        for t in range(gs):
            groups = {}
            for slot, o in mine[t]:
                groups.setdefault((o[tgt[0]], o[tgt[1]]), []).append((slot, o[dmp[0]], o[dmp[1]]))
            L += [f'{ind}case {t}: {{']
            for (a, b), terms in groups.items():
                expr = ' + '.join(f'gout[{s}]*dm[({da}+{x})*nao+({db}+{y})]' for s, x, y in terms)
                L += [f'{ind}    atomicAdd(vk+({ta}+{a})*nao+({tb}+{b}), {expr});']
            L += [f'{ind}    break; }}']
        L += [f'{ind}}}']
        if c == 3:
            L += [f'{I4}}}']
    L += [f'{I3}}}', f'{I2}}}', '    }', '}', '}', '']
    smem_max = nsq * (6 + 3 * G + 2 * 2 * ((li + lj + lk + ll) // 2 + 1))
    return name, G, per, '\n'.join(L), smem_max


PROLOGUE = '''while (1) {
    __syncthreads();
    if (t_id == 0) {
        int task_id = atomicAdd(head, 1);
        int batch_kl = task_id / bounds.npairs_ij;
        pair_ij = task_id - bounds.npairs_ij * batch_kl;
        pair_kl0 = batch_kl * (QUEUE_DEPTH - 512);
        uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
    }
    __syncthreads();
    if (pair_kl0 >= bounds.npairs_kl) {
        break;
    }
    if (jk.lr_factor != 0) {
        _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                       q_cond_ij, q_cond_kl, dm_penalty,
                       (int *)shared_memory, envs, bounds);
    } else {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          q_cond_ij, q_cond_kl, dm_penalty,
                          s_cond_ij, s_cond_kl, diffuse_exps,
                          (int *)shared_memory, jk.omega, envs, bounds);
    }
    if (ntasks == 0) {
        continue;
    }
    if (t_id == 0) {
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }
    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        __syncthreads();
        int iprim = bounds.iprim;
        int jprim = bounds.jprim;
        int kprim = bounds.kprim;
        int lprim = bounds.lprim;
        uint32_t bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            rlrk[0*nsq_per_block] = xlxk;
            rlrk[1*nsq_per_block] = ylyk;
            rlrk[2*nsq_per_block] = zlzk;
        }'''

PRIM_LOOPS = '''        for (int klp = 0; klp < kprim*lprim; ++klp) {
            __syncthreads();
            int kp = klp / lprim;
            int lp = klp % lprim;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            double ak = env[expk+kp];
            double al = env[expl+lp];
            double akl = ak + al;
            double al_akl = al / akl;
            if (gout_id == 0) {
                int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
                int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                gx[0] = fac_sym * ckcl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                if (fabs(cicj_cache[ijp]) < PRIM_CUTOFF) continue;
                __syncthreads();
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xij = ri[0] + rjri[0] * aj_aij;
                double yij = ri[1] + rjri[1] * aj_aij;
                double zij = ri[2] + rjri[2] * aj_aij;
                double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                int nroots = bounds.nroots;
                rys_roots_for_k(nroots, theta, rr, rw, jk.omega, jk.lr_factor, jk.sr_factor, nsq_per_block, ROOTS_STRIDE, gout_id);
                if (gout_id == 0) {
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    double cicj = cicj_cache[ijp];
                    gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                }'''

HEADER = '''// Generated by bench/gpu/gen_rys_k.py of the my-pyscf workspace: multi-thread Rys K kernels in the style of
// unrolled_rys_k.cu for classes that otherwise go to the generic rys_k_kernel. Do not edit by hand.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "gvhf-rys/create_tasks.cu"
#include "gvhf-rys/unrolled_kernels.cuh"

'''


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    kernels, cases = [], []
    for spec in specs:
        cls, _, shape = spec.partition(':')
        li, lj, lk, ll = (int(c) for c in cls)
        nsq, gs = (int(x) for x in (shape or '64x4').split('x'))
        assert nsq in (32, 64) and nsq * gs <= 256, spec
        name, G, per, text, smem = kernel(li, lj, lk, ll, nsq, gs)
        kernels.append(text)
        cases.append((li * 125 + lj * 25 + lk * 5 + ll, name, nsq, gs, G))
        print(f'{name}: {nsq}x{gs}, {per} outputs per thread, shared memory up to {smem * 8} bytes', file=sys.stderr)
    disp = ['int rys_k_generated(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,',
            '                    float *q_cond_ij, float *q_cond_kl, float dm_penalty,',
            '                    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,',
            '                    uint32_t *pool, int *head, int workers)', '{',
            '    int ijkl = bounds->li*125 + bounds->lj*25 + bounds->lk*5 + bounds->ll;',
            '    int nroots = bounds->nroots;', '    int ijprim = bounds->iprim * bounds->jprim;', '    switch (ijkl) {']
    for ijkl, name, nsq, gs, G in cases:
        disp += [f'    case {ijkl}: {{', f'        dim3 threads({nsq}, {gs});',
                 f'        int buflen = {nsq} * ({3 * G} + nroots*2 + 6) + ijprim;',
                 '        if (buflen*sizeof(double) > 48000) {',
                 f'            cudaFuncSetAttribute({name}, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));',
                 '        }', f'        LAUNCH_JKMATRIX_KERNEL({name});', '        return 1; }']
    disp += ['    }', '    return 0;', '}', '']
    open(out, 'w').write(HEADER + '\n'.join(kernels) + '\n' + '\n'.join(disp))


if __name__ == '__main__':
    main()
