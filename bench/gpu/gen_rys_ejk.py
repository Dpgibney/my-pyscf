"""Generator of multi-thread Rys gradient kernels for GPU4PySCF (gpu4pyscf/lib/gvhf-rys/generated_ejk_ip1.cu): the
two-electron energy first derivatives per atom of RYS_per_atom_jk_ip1.

GPU4PySCF's multi-thread gradient kernels (rys_ejk_ip1_1111, ..., rys_ejk_ip1_2210 in unrolled_ejk_ip1.cu) and its
generic rys_ejk_ip1_kernel compute, for every output (ij|kl) of a quartet and every root, twelve derivative terms
    v_ix += (2 ai g_x[t+1] - ix g_x[t-1]) * Iy Iz dd,  v_jx += ...,  v_kx, v_lx, and the same for y and z,
about 42 FP64 operations per output. Each derivative factor depends on one direction's index tuple only,
t = (ix, jx, kx, lx) for x, so grouping the outputs by tuple, one direction at a time, gives the same sums as
    sum_t f_x(t) S_x(t),   S_x(t) = sum over the outputs n with x tuple t of dd_n Iy(n) Iz(n).
Each output then costs 2 FP64 operations per direction, and each tuple of a direction 4 derivative factors and 4 FMAs.
These kernels do this in three passes per root, one per direction. In each pass the outputs, ordered by tuple, are cut
into gs runs of about equal cost, one per thread, so a tuple is split between at most two threads.

Everything else follows the unrolled multi-thread kernels:
  - nsq quartets per block (32 or 64, so that a warp holds one gout_id and the switch over gout_id never diverges) and
    gs threads per quartet; the primitive loops and the bra-pair skip;
  - the density products dd of every output, filled per task into dd_pool (nf * nsq per block, at most 101250, the
    DD_CACHE_MAX of 48 KB cards); a barrier at the top of each task keeps the next task's fill from overwriting the
    products another thread still reads;
  - threads 0-2 run the TRR and HRR of one direction each (gen_rys_k.recursions for (li+1, lj, lk+1, ll): the gradient
    needs i+1 and k+1); the j and l derivatives come from i+1 and k+1 by the HRR, as in the unrolled kernels;
  - each thread adds its sums with atomicAdd: per task to the atoms of k and l, per bra pair to those of i and j.
--files N writes the kernels into N translation units (OUT.cu with the dispatcher, OUT_1.cu, ...), balanced by size,
so that the library compiles in parallel.
Usage: gen_rys_ejk.py OUT.cu [--files N] CLASS[:NSQxGS] ...   e.g. gen_rys_ejk.py generated_ejk_ip1.cu 2111:32x8"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_rys_k import cart, recursions  # noqa: E402

DD_CACHE_MAX = 101250
XYZ = 'xyz'


def passes(li, lj, lk, ll, gs, k=1):
    '''For each direction, the runs of the gs // k warps. With k == 1 (a warp per gout_id) a run is a list of
    (tuple, [output index n]) chunks; with k > 1 (k gout_ids per warp) it is k lanes of (tuple, n) positions of equal
    length, padded with (None, nf)'''
    ci, cj, ck, cl = cart(li), cart(lj), cart(lk), cart(ll)
    outs = [(i, j, k, l) for l in range(len(cl)) for k in range(len(ck)) for j in range(len(cj)) for i in range(len(ci))]
    comps = [(ci[i], cj[j], ck[k], cl[l]) for i, j, k, l in outs]
    runs = []
    for d in range(3):
        groups = {}
        for n, c in enumerate(comps):
            groups.setdefault(tuple(x[d] for x in c), []).append(n)
        seq = [(t, n) for t in sorted(groups) for n in groups[t]]
        # cost: 2 FP64 per output, about 14 per chunk (4 derivative factors, 4 FMAs); cut at equal shares
        cum, acc = [], 0
        for p, (t, n) in enumerate(seq):
            acc += 2 + (14 if p + 1 == len(seq) or seq[p + 1][0] != t else 0)
            cum.append(acc)
        W = gs // k
        bounds = [0]
        for m in range(1, W):
            p = next(p for p, c in enumerate(cum) if c >= acc * m / W)
            bounds.append(max(p + 1, bounds[-1]))
        bounds.append(len(seq))
        dir_runs = []
        for m in range(W):
            if k > 1:
                run = seq[bounds[m]:bounds[m + 1]]
                n_pos = -(-len(run) // k)
                dir_runs.append([run[h * n_pos:(h + 1) * n_pos] + [(None, len(comps))] * (n_pos - len(run[h * n_pos:(h + 1) * n_pos]))
                                 for h in range(k)])
                continue
            chunks = []
            for t, n in seq[bounds[m]:bounds[m + 1]]:
                if chunks and chunks[-1][0] == t:
                    chunks[-1][1].append(n)
                else:
                    chunks.append((t, [n]))
            dir_runs.append(chunks)
        runs.append(dir_runs)
    return comps, runs


def kernel(li, lj, lk, ll, nsq, gs):
    name = f'rys_ejk_ip1_gen_{li}{lj}{lk}{ll}'
    nfi, nfj, nfk = len(cart(li)), len(cart(lj)), len(cart(lk))
    sj = li + 2
    sk = sj * (lj + 1)
    sl = sk * (lk + 2)
    G = sl * (ll + 1)
    k = max(1, 32 // nsq)   # gout_ids per warp
    comps, runs = passes(li, lj, lk, ll, gs, k)
    nf = len(comps)
    nfz = nf + (k > 1)      # a zero slot for the padding of the lanes
    assert nfz * nsq <= DD_CACHE_MAX, (name, nfz * nsq)
    # 64 KB per block on sm_75 for the short-range case (twice the roots), up to 64 primitive pairs
    nroots_sr = 2 * ((li + lj + lk + ll + 1) // 2 + 1)
    assert nsq * (6 + 3 * G + 2 * nroots_sr) * 8 + 64 * 8 + 128 <= 65536, (name, 'shared memory')

    def addr(c, d):
        return c[0][d] + c[1][d] * sj + c[2][d] * sk + c[3][d] * sl

    def g(d, a):
        return f'vgx[{(d * G + a) * nsq}]'
    I3, I4, I5 = ' ' * 12, ' ' * 16, ' ' * 20
    L = [f'// ({"spdfghi"[li]}{"spdfghi"[lj]}|{"spdfghi"[lk]}{"spdfghi"[ll]}): {nf} outputs, {gs} threads x {nsq} '
         f'quartets per block, g_size {G}', '__global__ static',
         f'__launch_bounds__({nsq * gs}, 1)', f'void {name}(JKENERGY_KERNEL_ARGS)', '{',
         '    JKENERGY_KERNEL_SETUP();', f'    constexpr int nsq_per_block = {nsq};', f'    constexpr int gout_stride = {gs};',
         '    int t_id = gout_id * nsq_per_block + sq_id;', '    int threads = nsq_per_block * gout_stride;',
         '    int nbas = envs.nbas;', '    int *bas = envs.bas;', '    double *env = envs.env;',
         '    int iprim = bounds.iprim;', '    int jprim = bounds.jprim;', '    int kprim = bounds.kprim;',
         '    int lprim = bounds.lprim;', '    int nroots = bounds.nroots;', '    double *rlrk = shared_memory + sq_id;',
         f'    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;',
         f'    double *gx = shared_memory + nsq_per_block * 6 + sq_id;',
         f'    double *rw = shared_memory + nsq_per_block * {6 + 3 * G} + sq_id;',
         f'    double *cicj_cache = shared_memory + nsq_per_block * ({6 + 3 * G}+nroots*2);', '',
         '    uint32_t *bas_kl_idx = pool + worker_id * QUEUE_DEPTH;',
         f'    double *dd_cache = dd_pool + worker_id * {nfz * nsq} + sq_id;']
    L += PROLOGUE.replace('NSQ', str(nsq)).splitlines()
    # density products of the task's outputs, n = i + nfi*(j + nfj*(k + nfk*l))
    for n_dm, dms in ((1, None), (2, 'dmb')):
        L += [f'        {"if (jk.n_dm == 1) {" if n_dm == 1 else "} else {"}']
        if n_dm == 2:
            L += ['            double *dmb = dm + nao * nao;']
        L += [f'            for (int n = gout_id; n < {nf}; n+={gs}) {{',
              f'                int i = n % {nfi};', f'                int j = n / {nfi} % {nfj};',
              f'                int k = n / {nfi * nfj} % {nfk};', f'                int l = n / {nfi * nfj * nfk};',
              '                int _i = i + i0;', '                int _j = j + j0;', '                int _k = k + k0;',
              '                int _l = l + l0;', '                int _jl = _j*nao+_l;', '                int _jk = _j*nao+_k;',
              '                int _li = _l*nao+_i;', '                int _ki = _k*nao+_i;', '                int _ji = _j*nao+_i;',
              '                int _lk = _l*nao+_k;', '                double dd = 0;', '                if (do_k) {']
        if n_dm == 1:
            L += ['                    dd += jk.k_factor * (dm[_jk] * dm[_li] + dm[_jl] * dm[_ki]);', '                }',
                  '                if (do_j) {', '                    dd += jk.j_factor * dm[_ji] * dm[_lk];', '                }']
        else:
            L += ['                    dd += jk.k_factor * (dm [_jk] * dm [_li] + dm [_jl] * dm [_ki] +',
                  '                                         dmb[_jk] * dmb[_li] + dmb[_jl] * dmb[_ki]);', '                }',
                  '                if (do_j) {',
                  '                    dd += jk.j_factor * (dm[_ji] + dmb[_ji]) * (dm[_lk] + dmb[_lk]);', '                }']
        L += [f'                dd_cache[n*{nsq}] = fac_sym * dd;', '            }']
    L += ['        }']
    if k > 1:
        L += ['        if (gout_id == 0) {', f'            dd_cache[{nf * nsq}] = 0;', '        }']
    L += PRIM_LOOPS.replace('NSQ', str(nsq)).replace('GSIZE', str(G)).replace('GS', str(gs)).splitlines()
    # root loop: TRR and HRR, one direction per thread
    lij, lkl = li + 1 + lj, lk + 1 + ll
    body = [f'{I4}for (int irys = 0; irys < nroots; ++irys) {{', f'{I4}    __syncthreads();', f'{I4}    double s0, s1, s2;',
            f'{I4}    double rt = rw[irys*{2 * nsq}];', f'{I4}    double rt_aa = rt / (aij + akl);',
            f'{I4}    double rt_aij = rt_aa * akl;', f'{I4}    double rt_akl = rt_aa * aij;']
    if lij > 1:
        body += [f'{I4}    double b10 = .5/aij * (1 - rt_aij);']
    body += [f'{I4}    double b00 = .5 * rt_aa;']
    if lkl > 1:
        body += [f'{I4}    double b01 = .5/akl * (1 - rt_akl);']
    body += [f'{I4}    for (int n = gout_id; n < 3; n += {gs}) {{', f'{I4}        if (n == 2) {{',
             f'{I4}            gx[{2 * G * nsq}] = rw[irys*{2 * nsq}+{nsq}];', f'{I4}        }}',
             f'{I4}        double *_gx = gx + n * {G * nsq};', f'{I4}        double xjxi = rjri[n];',
             f'{I4}        double Rpa = xjxi * aj_aij;', f'{I4}        double c0x = Rpa - rt_aij * Rpq[n*{nsq}];']
    body += [f'{I4}        {s}' for s in recursions(li + 1, lj, lk + 1, ll, nsq)]
    # volatile: otherwise the compiler keeps every value it loaded for a later run in registers (255 registers and
    # spills), where reloading from shared memory is cheaper
    body += [f'{I4}    }}', f'{I4}    __syncthreads();', f'{I4}    double s;', f'{I4}    volatile double *vgx = gx;']
    rj = ['xjxi', 'yjyi', 'zjzi']
    rl = ['xlxk', 'ylyk', 'zlzk']
    if k > 1:
        body += merged(runs, comps, k, nsq, G, sj, sk, sl, addr, I4, I5)
        gs_loop = 0
    else:
        body += [f'{I4}    switch (gout_id) {{']
        gs_loop = gs
    for m in range(gs_loop):
        body += [f'{I4}    case {m}:']
        for d in range(3):
            d1, d2 = [e for e in range(3) if e != d]
            for t, ns in runs[d][m]:
                for p, n in enumerate(ns):
                    c = comps[n]
                    prod = f'dd_cache[{n * nsq}] * ({g(d1, addr(c, d1))} * {g(d2, addr(c, d2))})'
                    body += [f'{I5}s {"=" if p == 0 else "+="} {prod};']
                ix, jx, kx, lx = t
                a = ix + jx * sj + kx * sk + lx * sl
                fi = f'ai2 * {g(d, a + 1)}' + (f' - {ix} * {g(d, a - 1)}' if ix else '')
                fj = f'aj2 * ({g(d, a + 1)} - {rj[d]} * {g(d, a)})' + (f' - {jx} * {g(d, a - sj)}' if jx else '')
                fk = f'ak2 * {g(d, a + sk)}' + (f' - {kx} * {g(d, a - sk)}' if kx else '')
                fl = f'al2 * ({g(d, a + sk)} - {rl[d]} * {g(d, a)})' + (f' - {lx} * {g(d, a - sl)}' if lx else '')
                for v, f in (('i', fi), ('j', fj), ('k', fk), ('l', fl)):
                    body += [f'{I5}v_{v}{XYZ[d]} += ({f}) * s;']
        body += [f'{I5}break;']
    body += [f'{I4}    }}', f'{I4}}}']
    L += body + [f'{I3}}}', '        }']
    L += EPILOGUE.splitlines()
    smem_max = nsq * (6 + 3 * G + 2 * nroots_sr)
    if k > 1:
        ops = [sum(2 * len(lanes[0]) for d in range(3) for lanes in [runs[d][w]]) for w in range(gs // k)]
    else:
        ops = [sum(2 * len(ns) + 14 for d in range(3) for t, ns in runs[d][m]) for m in range(gs)]
    return name, G, nf, '\n'.join(L), smem_max, ops


def kernel_one_thread(li, lj, lk, ll, nsq):
    '''One quartet per thread, as GPU4PySCF's one-thread gradient kernels: no barrier in the task loop, the density
    products and the g arrays in registers (constant indices into local arrays), the same sums per tuple'''
    name = f'rys_ejk_ip1_gen_{li}{lj}{lk}{ll}'
    sj = li + 2
    sk = sj * (lj + 1)
    sl = sk * (lk + 2)
    G = sl * (ll + 1)
    comps, runs = passes(li, lj, lk, ll, 1)
    nf = len(comps)
    ci, cj, ck, cl = cart(li), cart(lj), cart(lk), cart(ll)
    nfi, nfj, nfk = len(ci), len(cj), len(ck)

    def addr(c, d):
        return c[0][d] + c[1][d] * sj + c[2][d] * sk + c[3][d] * sl

    def g(d, a):
        return f'g[{d * G + a}]'
    I3, I4, I5 = ' ' * 12, ' ' * 16, ' ' * 20
    L = [f'// ({"spdfghi"[li]}{"spdfghi"[lj]}|{"spdfghi"[lk]}{"spdfghi"[ll]}): {nf} outputs, one thread per quartet, '
         f'g_size {G}', '__global__ static', f'void {name}(JKENERGY_KERNEL_ARGS)', '{', '    JKENERGY_KERNEL_SETUP();',
         f'    constexpr int nsq_per_block = {nsq};', '    constexpr int gout_stride = 1;',
         '    int t_id = gout_id * nsq_per_block + sq_id;', '    int threads = nsq_per_block * gout_stride;',
         '    int nbas = envs.nbas;', '    int *bas = envs.bas;', '    double *env = envs.env;',
         '    int iprim = bounds.iprim;', '    int jprim = bounds.jprim;', '    int kprim = bounds.kprim;',
         '    int lprim = bounds.lprim;', '    int nroots = bounds.nroots;', '    double *rw = shared_memory + sq_id;',
         '    double *cicj_cache = shared_memory + nsq_per_block * (nroots*2);', '',
         '    uint32_t *bas_kl_idx = pool + worker_id * QUEUE_DEPTH;']
    pro = PROLOGUE.replace('NSQ', str(nsq)).splitlines()
    # the one-thread kernels leave the task loop at the first padding task and have no shared rlrk
    i = pro.index('        // another thread of this quartet may still read the previous task\'s dd_cache')
    pro[i:i + 2] = ['        if (task_id >= ntasks) break;']
    j = pro.index('        if (gout_id == 0) {')
    del pro[j:j + 5]
    L += pro
    L += [f'        double dd[{nf}];', '        if (jk.n_dm == 1) {', '            double *dmb = dm;']
    for n_dm in (1, 2):
        if n_dm == 2:
            L += ['        } else {', '            double *dmb = dm + nao * nao;']
        for n, (i_, j_, k_, l_) in enumerate((i, j, k, l) for l in range(len(cl)) for k in range(len(ck))
                                             for j in range(len(cj)) for i in range(nfi)):
            I, J, K, Lx = f'(i0+{i_})', f'(j0+{j_})', f'(k0+{k_})', f'(l0+{l_})'
            if n_dm == 1:
                kk = f'dm[{J}*nao+{K}] * dm[{Lx}*nao+{I}] + dm[{J}*nao+{Lx}] * dm[{K}*nao+{I}]'
                jj = f'dm[{J}*nao+{I}] * dm[{Lx}*nao+{K}]'
            else:
                kk = (f'dm[{J}*nao+{K}] * dm[{Lx}*nao+{I}] + dm[{J}*nao+{Lx}] * dm[{K}*nao+{I}] + '
                      f'dmb[{J}*nao+{K}] * dmb[{Lx}*nao+{I}] + dmb[{J}*nao+{Lx}] * dmb[{K}*nao+{I}]')
                jj = f'(dm[{J}*nao+{I}] + dmb[{J}*nao+{I}]) * (dm[{Lx}*nao+{K}] + dmb[{Lx}*nao+{K}])'
            L += [f'            dd[{n}] = fac_sym * ((do_k ? jk.k_factor * ({kk}) : 0.) + '
                  f'(do_j ? jk.j_factor * {jj} : 0.));']
    L += ['        }']
    L += ['        for (int klp = 0; klp < kprim*lprim; ++klp) {', '            int kp = klp / lprim;',
          '            int lp = klp % lprim;', '            double ak = env[expk+kp];', '            double al = env[expl+lp];',
          '            double ak2 = ak * 2;', '            double al2 = al * 2;', '            double akl = ak + al;',
          '            double al_akl = al / akl;', '            double theta_kl = ak * al_akl;',
          '            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));',
          '            double ckcl = env[ck+kp] * env[cl+lp] * Kcd;', '            if (fabs(ckcl) < PRIM_CUTOFF) continue;',
          '            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {', '                int ip = ijp / jprim;',
          '                int jp = ijp % jprim;', '                double ai = env[expi+ip];', '                double aj = env[expj+jp];',
          '                double ai2 = ai * 2;', '                double aj2 = aj * 2;', '                double aij = ai + aj;',
          '                double aj_aij = aj / aij;', '                double cicj = cicj_cache[ijp];',
          '                if (fabs(cicj) < PRIM_CUTOFF) continue;',
          '                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));',
          '                double xij = ri[0] + rjri[0] * aj_aij;', '                double yij = ri[1] + rjri[1] * aj_aij;',
          '                double zij = ri[2] + rjri[2] * aj_aij;', '                double xkl = env[rk+0] + xlxk * al_akl;',
          '                double ykl = env[rk+1] + ylyk * al_akl;', '                double zkl = env[rk+2] + zlzk * al_akl;',
          '                double Rpq[3] = {xij - xkl, yij - ykl, zij - zkl};',
          '                double rlrk[3] = {xlxk, ylyk, zlzk};',
          '                double rr = Rpq[0]*Rpq[0] + Rpq[1]*Rpq[1] + Rpq[2]*Rpq[2];',
          '                double theta = aij * akl / (aij + akl);',
          '                rys_roots_for_k(nroots, theta, rr, rw, jk.omega, jk.lr_factor, jk.sr_factor, nsq_per_block, 1, 0);']
    lij, lkl = li + 1 + lj, lk + 1 + ll
    body = [f'{I4}for (int irys = 0; irys < nroots; ++irys) {{', f'{I4}    double s0, s1, s2;',
            f'{I4}    double rt = rw[irys*{2 * nsq}];', f'{I4}    double rt_aa = rt / (aij + akl);',
            f'{I4}    double rt_aij = rt_aa * akl;', f'{I4}    double rt_akl = rt_aa * aij;']
    if lij > 1:
        body += [f'{I4}    double b10 = .5/aij * (1 - rt_aij);']
    body += [f'{I4}    double b00 = .5 * rt_aa;']
    if lkl > 1:
        body += [f'{I4}    double b01 = .5/akl * (1 - rt_akl);']
    body += [f'{I4}    double g[{3 * G}];', f'{I4}    g[0] = 1.;', f'{I4}    g[{G}] = 1.;',
             f'{I4}    g[{2 * G}] = rw[irys*{2 * nsq}+{nsq}] * fac;']
    for d in range(3):
        body += [f'{I4}    {{', f'{I4}        constexpr int n = {d};', f'{I4}        double *_gx = g + {d * G};',
                 f'{I4}        double xjxi = rjri[n];', f'{I4}        double Rpa = xjxi * aj_aij;',
                 f'{I4}        double c0x = Rpa - rt_aij * Rpq[n];']
        body += [f'{I4}        {s}' for s in recursions(li + 1, lj, lk + 1, ll, 1)]
        body += [f'{I4}    }}']
    rj = ['xjxi', 'yjyi', 'zjzi']
    rl = ['xlxk', 'ylyk', 'zlzk']
    body += [f'{I4}    double s;']
    for d in range(3):
        d1, d2 = [e for e in range(3) if e != d]
        for t, ns in runs[d][0]:
            for p, n in enumerate(ns):
                c = comps[n]
                body += [f'{I4}    s {"=" if p == 0 else "+="} dd[{n}] * ({g(d1, addr(c, d1))} * {g(d2, addr(c, d2))});']
            ix, jx, kx, lx = t
            a = ix + jx * sj + kx * sk + lx * sl
            fi = f'ai2 * {g(d, a + 1)}' + (f' - {ix} * {g(d, a - 1)}' if ix else '')
            fj = f'aj2 * ({g(d, a + 1)} - {rj[d]} * {g(d, a)})' + (f' - {jx} * {g(d, a - sj)}' if jx else '')
            fk = f'ak2 * {g(d, a + sk)}' + (f' - {kx} * {g(d, a - sk)}' if kx else '')
            fl = f'al2 * ({g(d, a + sk)} - {rl[d]} * {g(d, a)})' + (f' - {lx} * {g(d, a - sl)}' if lx else '')
            for v, f in (('i', fi), ('j', fj), ('k', fk), ('l', fl)):
                body += [f'{I4}    v_{v}{XYZ[d]} += ({f}) * s;']
    body += [f'{I4}}}']
    L += body + [f'{I3}}}', '        }']
    L += EPILOGUE.splitlines()
    return name, G, nf, '\n'.join(L), nsq * 2 * 2 * ((li + lj + lk + ll + 1) // 2 + 1), [0]


def sel(vals):
    '''The value of lane h (h = gout_id % k) as a C expression of the per-lane constants'''
    if all(v == vals[0] for v in vals):
        return str(vals[0])
    half = len(vals) // 2
    return f'(h & {half} ? {sel(vals[half:])} : {sel(vals[:half])})'


def merged(runs, comps, k, nsq, G, sj, sk, sl, addr, I4, I5):
    '''k gout_ids share a warp: one instruction stream per warp, each lane with its own addresses (selected by h), so
    the warp does not diverge. All lanes apply their sums wherever one lane's tuple changes.'''
    rj = ['xjxi', 'yjyi', 'zjzi']
    rl = ['xlxk', 'ylyk', 'zlzk']
    nf = len(comps)

    def g(d, vals):
        return f'vgx[{sel([(d * G + a) * nsq for a in vals])}]'
    # h is read inside the root loop through asm volatile: otherwise the compiler hoists every selected address out
    # of all loops (they depend on h only) and keeps hundreds of them live
    out = [f'{I4}    unsigned tid_y;', f'{I4}    int h;', f'{I4}    switch (gout_id / {k}) {{']
    for w in range(len(runs[0])):
        out += [f'{I4}    case {w}:']
        for d in range(3):
            d1, d2 = [e for e in range(3) if e != d]
            # once per pass, so that a pass's selected addresses are computed in that pass
            out += [f'{I5}asm volatile("mov.u32 %0, %%tid.y;" : "=r"(tid_y));', f'{I5}h = tid_y % {k};']
            lanes = runs[d][w]
            last = [None] * k
            fresh = True
            for p in range(len(lanes[0])):
                pos = [lane[p] for lane in lanes]
                ns = [n for t, n in pos]
                cs = [comps[n] if n < nf else comps[0] for n in ns]
                last = [t if t is not None else last[h] for h, (t, n) in enumerate(pos)]
                prod = (f'dd_cache[{sel([n * nsq for n in ns])}] * ({g(d1, [addr(c, d1) for c in cs])} * '
                        f'{g(d2, [addr(c, d2) for c in cs])})')
                out += [f'{I5}s {"=" if fresh else "+="} {prod};']
                ends = p + 1 == len(lanes[0]) or any(
                    lane[p + 1][0] is not None and lane[p + 1][0] != last[h] for h, lane in enumerate(lanes))
                fresh = ends
                if not ends:
                    continue
                ts = [t if t is not None else (0, 0, 0, 0) for t in last]
                a = [t[0] + t[1] * sj + t[2] * sk + t[3] * sl for t in ts]

                def coef(i):
                    return sel([f'{float(t[i])}' for t in ts])

                def minus(i, step):
                    if all(t[i] == 0 for t in ts):
                        return ''
                    return f' - {coef(i)} * {g(d, [x - step if t[i] else x for x, t in zip(a, ts)])}'
                fi = f'ai2 * {g(d, [x + 1 for x in a])}' + minus(0, 1)
                fj = f'aj2 * ({g(d, [x + 1 for x in a])} - {rj[d]} * {g(d, a)})' + minus(1, sj)
                fk = f'ak2 * {g(d, [x + sk for x in a])}' + minus(2, sk)
                fl = f'al2 * ({g(d, [x + sk for x in a])} - {rl[d]} * {g(d, a)})' + minus(3, sl)
                for v, f in (('i', fi), ('j', fj), ('k', fk), ('l', fl)):
                    out += [f'{I5}v_{v}{XYZ[d]} += ({f}) * s;']
        out += [f'{I5}break;']
    return out


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
        _fill_ejk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                        q_cond_ij, q_cond_kl,
                        (int *)shared_memory, jk, envs, bounds);
    } else {
        _fill_sr_ejk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl,
                           s_cond_ij, s_cond_kl, diffuse_exps,
                           (int *)shared_memory, jk, envs, bounds);
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
    __syncthreads();

    int *ao_loc = envs.ao_loc;
    int nao = ao_loc[nbas];
    double *dm = jk.dm;
    double v_ix = 0;
    double v_iy = 0;
    double v_iz = 0;
    double v_jx = 0;
    double v_jy = 0;
    double v_jz = 0;
    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        // another thread of this quartet may still read the previous task's dd_cache
        __syncthreads();
        uint32_t bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        double fac_sym = PI_FAC;
        if (task_id < ntasks) {
            if (ish == jsh) fac_sym *= .5;
            if (ksh == lsh) fac_sym *= .5;
            if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
        } else {
            fac_sym = 0;
        }
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        double xlxk = env[rl+0] - env[rk+0];
        double ylyk = env[rl+1] - env[rk+1];
        double zlzk = env[rl+2] - env[rk+2];
        if (gout_id == 0) {
            rlrk[0] = xlxk;
            rlrk[NSQ] = ylyk;
            rlrk[2*NSQ] = zlzk;
        }
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        int do_j = jk.j_factor != 0.;
        int do_k = jk.k_factor != 0.;'''

PRIM_LOOPS = '''        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = env[expk+kp];
            double al = env[expl+lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
            double akl = ak + al;
            double al_akl = al / akl;
            double theta_kl = ak * al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                gx[0] = ckcl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                if (fabs(cicj_cache[ijp]) < PRIM_CUTOFF) continue;
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xij = ri[0] + rjri[0] * aj_aij;
                double yij = ri[1] + rjri[1] * aj_aij;
                double zij = ri[2] + rjri[2] * aj_aij;
                double xkl = env[rk+0] + xlxk * al_akl;
                double ykl = env[rk+1] + ylyk * al_akl;
                double zkl = env[rk+2] + zlzk * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                __syncthreads();
                if (gout_id == 0) {
                    Rpq[0] = xpq;
                    Rpq[NSQ] = ypq;
                    Rpq[2*NSQ] = zpq;
                    double cicj = cicj_cache[ijp];
                    gx[GSIZE*NSQ] = cicj / (aij*akl*sqrt(aij+akl));
                }
                rys_roots_for_k(nroots, theta, rr, rw, jk.omega, jk.lr_factor, jk.sr_factor, nsq_per_block, GS, gout_id);'''

EPILOGUE = '''        if (task_id >= ntasks) {
            break;
        }
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
    int ia = bas[ish*BAS_SLOTS+ATOM_OF];
    int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
    double *ejk = jk.ejk;
    atomicAdd(ejk+ia*3+0, v_ix);
    atomicAdd(ejk+ia*3+1, v_iy);
    atomicAdd(ejk+ia*3+2, v_iz);
    atomicAdd(ejk+ja*3+0, v_jx);
    atomicAdd(ejk+ja*3+1, v_jy);
    atomicAdd(ejk+ja*3+2, v_jz);
}
}
'''

HEADER = '''// Generated by bench/gpu/gen_rys_ejk.py of the my-pyscf workspace: multi-thread Rys gradient kernels
// (RYS_per_atom_jk_ip1) that sum the outputs per Cartesian tuple before applying the derivative factors.
// Do not edit by hand.
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "gvhf-rys/create_tasks.cu"
#include "gvhf-rys/unrolled_kernels.cuh"

'''


def dispatcher(decl, cases):
    pad = ' ' * (len(decl) + 1)
    disp = [f'{decl}(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,',
            f'{pad}float *q_cond_ij, float *q_cond_kl, float dm_penalty,',
            f'{pad}float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,',
            f'{pad}uint32_t *pool, double *dd_pool, int *head, int workers)', '{',
            '    int ijkl = bounds->li*125 + bounds->lj*25 + bounds->lk*5 + bounds->ll;',
            '    int nroots = bounds->nroots;', '    int ijprim = bounds->iprim * bounds->jprim;', '    switch (ijkl) {']
    for ijkl, kname, nsq, gs, G in cases:
        if gs == 1:
            disp += [f'    case {ijkl}: {{', f'        int nsq_per_block = {nsq};', f'        dim3 threads({nsq}, 1);',
                     f'        int buflen = nroots*2 * {nsq} + ijprim;',
                     f'        LAUNCH_JKENERGY_KERNEL_PERSISTENT({kname});', '        return 1; }']
            continue
        disp += [f'    case {ijkl}: {{', f'        dim3 threads({nsq}, {gs});',
                 f'        int buflen = {nsq} * ({3 * G} + nroots*2 + 6) + ijprim;',
                 '        if (buflen*sizeof(double) > 48000) {',
                 f'            cudaFuncSetAttribute({kname}, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));',
                 '        }', f'        LAUNCH_JKENERGY_KERNEL({kname});', '        return 1; }']
    return disp + ['    }', '    return 0;', '}', '']


ARGS = ('RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds, float *q_cond_ij, float *q_cond_kl, '
        'float dm_penalty, float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, uint32_t *pool, double *dd_pool, '
        'int *head, int workers')
CALL = 'envs, jk, bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head, workers'


def main():
    args = sys.argv[1:]
    n_files = 1
    if args[1] == '--files':
        n_files = int(args[2])
        args = args[:1] + args[3:]
    out, specs = args[0], args[1:]
    built = []
    for spec in specs:
        cls, _, shape = spec.partition(':')
        li, lj, lk, ll = (int(c) for c in cls)
        nsq, gs = (int(x) for x in (shape or '32x8').split('x'))
        if gs == 1:
            assert nsq in (64, 128, 256), spec
            name, G, nf, text, smem, ops = kernel_one_thread(li, lj, lk, ll, nsq)
        else:
            assert nsq in (8, 16, 32, 64) and nsq * gs <= 512 and gs % max(1, 32 // nsq) == 0, spec
            name, G, nf, text, smem, ops = kernel(li, lj, lk, ll, nsq, gs)
        built.append((text, (li * 125 + lj * 25 + lk * 5 + ll, name, nsq, gs, G)))
        print(f'{name}: {nsq}x{gs}, {nf} outputs, shared memory up to {smem * 8} bytes, output-phase FP64 per thread '
              f'{min(ops)}-{max(ops)}', file=sys.stderr)
    # several translation units, balanced by size, so that the library compiles in parallel
    parts = [[] for _ in range(n_files)]
    for item in sorted(built, key=lambda b: -len(b[0])):
        min(parts, key=lambda q: sum(len(t) for t, c in q)).append(item)
    base = out[:-3]
    for f, part in enumerate(parts):
        text = HEADER + '\n'.join(t for t, c in part) + '\n'
        if n_files == 1:
            text += '\n'.join(dispatcher('int rys_ejk_ip1_generated', [c for t, c in part]))
        elif f == 0:
            text += ''.join(f'extern int rys_ejk_ip1_generated_{g}({ARGS});\n' for g in range(1, n_files))
            text += '\n'.join(dispatcher('static int rys_ejk_ip1_generated_0', [c for t, c in part]))
            text += f'int rys_ejk_ip1_generated({ARGS})\n{{\n    return ' + ' ||\n        '.join(
                f'rys_ejk_ip1_generated_{g}({CALL})' for g in range(n_files)) + ';\n}\n'
        else:
            text += '\n'.join(dispatcher(f'int rys_ejk_ip1_generated_{f}', [c for t, c in part]))
        open(out if f == 0 else f'{base}_{f}.cu', 'w').write(text)


if __name__ == '__main__':
    main()
