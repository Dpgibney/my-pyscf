"""Generator of multi-thread Rys kernels for the Hessian's two-electron second derivatives in GPU4PySCF
(gpu4pyscf/lib/gvhf-rys/generated_ejk_ip2.cu): RYS_per_atom_jk_ip2_type12, (d²i j|kl), (di dj|kl) and the same on
the ket, and RYS_per_atom_jk_ip2_type3, (di j|dk l) and the other three bra-ket pairs.

GPU4PySCF's generic kernels (rys_contract_jk_ip2.cu) build, for every output (ij|kl) of a quartet and every root, the
derivative factors of all three directions and add 42 (type12) or 36 (type3) terms: about 180 and 160 FP64 operations
per output (NOTES.md part 15). These kernels sum per Cartesian tuple, as gen_rys_ejk.py does for the gradient:
  - a term with both derivatives in one direction, such as F(t_x) Iy Iz dd, is F(t_x) S_x(t_x), with S_x(t) the sum of
    dd Iy Iz over the outputs whose x tuple (ix, jx, kx, lx) is t;
  - a term with the derivatives in two directions, such as fi(t_x) fj(t_y) Iz dd, is grouped by the x tuple:
    fi(t_x) T(t_x), T(t) the sum of fj(t_y) Iz dd over the outputs with x tuple t. The four first-derivative factors
    of the second direction (fi, fj, fk, fl) give four such sums, from which all eight terms of a pair of directions
    follow (type12: ij, ji, ii, jj and kl, lk, kk, ll; type3: ik, ki, jk, kj, il, li, jl, lj).
Three passes per root, by x tuples (with the pair x, y), by y (y, z) and by z (z, x): about 6 FP64 operations per
output and pass. The first-derivative factors of every tuple are computed once per root into a table in shared
memory (all threads of a quartet share the work), and the second-derivative factors at the end of each tuple's run.

Everything else follows gen_rys_ejk.py: nsq quartets per block and gs threads per quartet; the density products of a
task in dd_pool; threads 0-2 run the TRR and HRR of one direction each (gen_rys_k.recursions: type12 needs i, j, k and
l one higher, type3 i and k, as the generic kernels); the outputs of a pass, ordered by tuple, cut into runs of about
equal cost, one per thread; where a warp holds k > 1 gout_ids, its lanes run one instruction stream with per-lane
addresses (h = gout_id % k). The bra pair's sums (type12: ij, ii, jj) are added once per bra pair, the others per task.
--drop-l uses translational invariance: at a fixed root the first-derivative factors of a tuple sum to zero,
fi + fj + fk + fl = 0, so T3 = -(T0 + T1 + T2), formed at the end of each tuple's run instead of with an FMA per output.
--files N writes the kernels into N translation units, balanced by size.
Usage: gen_rys_ejk_ip2.py OUT.cu [--files N] [--drop-l] KIND:CLASS[:NSQxGS] ...   e.g. gen_rys_ejk_ip2.py out.cu 12:2110:16x16 3:2110"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_rys_k import cart, recursions  # noqa: E402
from gen_rys_ejk import sel, DD_CACHE_MAX, PRIM_LOOPS  # noqa: E402

XYZ = 'xyz'
COST_OUT, COST_TUPLE = 8, 60   # rough FP64 operations per output and per tuple end, for cutting the runs
DROP_L = False   # --drop-l
GROUPS = {12: (('ij', 'bra', 9), ('ii', 'bra', 6), ('jj', 'bra', 6), ('kl', 'task', 9), ('kk', 'task', 6),
               ('ll', 'task', 6)),
          3: (('ik', 'task', 9), ('jk', 'task', 9), ('il', 'task', 9), ('jl', 'task', 9))}


def layout(kind, li, lj, lk, ll):
    '''The g arrays of the generic kernels: type12 i, j, k and l one higher, type3 i and k'''
    rec = (li + 1, lj + 1, lk + 1, ll + 1) if kind == 12 else (li + 1, lj, lk + 1, ll)
    sj = rec[0] + 1
    sk = sj * (rec[1] + 1)
    sl = sk * (rec[2] + 1)
    return rec, sj, sk, sl, sl * (rec[3] + 1)


def tuples(li, lj, lk, ll):
    '''The index tuples of a direction, t = i + (li+1)*(j + (lj+1)*(k + (lk+1)*l))'''
    return [(i, j, k, l) for l in range(ll + 1) for k in range(lk + 1) for j in range(lj + 1) for i in range(li + 1)]


def var(grp, a, b):
    '''The sum of group grp (ij, ii, ... ik, ...) for the directions a, b; ii, jj, kk, ll are symmetric in a, b'''
    if grp[0] == grp[1] and a > b:
        a, b = b, a
    return f'v{grp}_{XYZ[a]}{XYZ[b]}'


def group_vars(kind, scope):
    out = []
    for grp, sc, n in GROUPS[kind]:
        if sc != scope:
            continue
        for a in range(3):
            for b in range(3):
                if grp[0] == grp[1] and a > b:
                    continue
                out.append(var(grp, a, b))
    return out


def passes(li, lj, lk, ll, gs, k):
    '''For each direction, the runs of the gs // k warp groups: k lanes of (tuple, n) positions of equal length,
    padded with (None, nf), over the outputs ordered by that direction's tuple'''
    ci, cj, ck, cl = cart(li), cart(lj), cart(lk), cart(ll)
    outs = [(i, j, k_, l) for l in range(len(cl)) for k_ in range(len(ck)) for j in range(len(cj))
            for i in range(len(ci))]
    comps = [(ci[i], cj[j], ck[k_], cl[l]) for i, j, k_, l in outs]
    W = gs // k
    runs = []
    for d in range(3):
        groups = {}
        for n, c in enumerate(comps):
            groups.setdefault(tuple(x[d] for x in c), []).append(n)
        seq = [(t, n) for t in sorted(groups) for n in groups[t]]
        cum, acc = [], 0
        for p, (t, n) in enumerate(seq):
            acc += COST_OUT - DROP_L + (COST_TUPLE if p + 1 == len(seq) or seq[p + 1][0] != t else 0)
            cum.append(acc)
        bounds = [0]
        for m in range(1, W):
            p = next(p for p, c in enumerate(cum) if c >= acc * m / W)
            bounds.append(max(p + 1, bounds[-1]))
        bounds.append(len(seq))
        dir_runs = []
        for m in range(W):
            run = seq[bounds[m]:bounds[m + 1]]
            n_pos = -(-len(run) // k)
            lanes = [run[h * n_pos:(h + 1) * n_pos] for h in range(k)]
            dir_runs.append([lane + [(None, len(comps))] * (n_pos - len(lane)) for lane in lanes])
        runs.append(dir_runs)
    return comps, runs


class Code:
    '''Expressions on the g arrays and the factor table of one quartet, with an address per lane'''
    def __init__(self, kind, li, lj, lk, ll, nsq):
        self.kind = kind
        self.nsq = nsq
        _, self.sj, self.sk, self.sl, self.G = layout(kind, li, lj, lk, ll)
        self.tups = tuples(li, lj, lk, ll)
        self.tid = {t: n for n, t in enumerate(self.tups)}
        self.NT = len(self.tups)

    def addr(self, t):
        return t[0] + t[1] * self.sj + t[2] * self.sk + t[3] * self.sl

    def g(self, d, addrs):
        return f'vgx[{sel([(d * self.G + a) * self.nsq for a in addrs])}]'

    def tab(self, d, ts, b):
        '''factor b (fi, fj, fk, fl) of direction d at the lanes' tuples'''
        return f'vgx[{sel([(3 * self.G + (d * self.NT + self.tid[t]) * 4 + b) * self.nsq for t in ts])}]'

    @staticmethod
    def coef(vals):
        return sel([f'{float(v)}' for v in vals])

    def second_factors(self, d, ts):
        '''The second-derivative factors of direction d at the lanes' tuples ts (type12: ij, ii, jj, kl, kk, ll;
        type3: ik, jk, il, jl), as in the generic kernels. A term that needs a lower index of a lane whose index is 0
        reads that lane's own g value (finite) with a coefficient 0.'''
        a = [self.addr(t) for t in ts]
        sj, sk, sl = self.sj, self.sk, self.sl
        rj, rl = ['xjxi', 'yjyi', 'zjzi'][d], ['xlxk', 'ylyk', 'zlzk'][d]

        def G(off, cond=None):
            '''g at a + off, or at a for the lanes where cond(t) is false'''
            return self.g(d, [x + off if cond is None or cond(t) else x for x, t in zip(a, ts)])
        c = self.coef

        def has(f):
            return any(f(t) for t in ts)
        I, J, K, L = (lambda t: t[0] > 0), (lambda t: t[1] > 0), (lambda t: t[2] > 0), (lambda t: t[3] > 0)
        out = {}
        if self.kind == 12:
            def same(a2, step, rr, sign, idx, c2):
                '''the second derivative of one center: a2 (a2 (g(+2) ...) - (2n+1) g) + n(n-1) g(-2)'''
                inc2 = f'({G(c2)} {sign} {rr} * {G(step)})'
                e = f'{a2} * ({a2} * {inc2} - {c([2 * t[idx] + 1 for t in ts])} * {G(0)})'
                if has(lambda t: t[idx] > 1):
                    e += f' + {c([t[idx] * (t[idx] - 1) for t in ts])} * {G(-2 * [1, sj, sk, sl][idx], lambda t: t[idx] > 1)}'
                return e

            def mixed(a1, s1, n1, a2, s2, n2):
                '''the mixed derivative of two centers 1, 2 (steps s1, s2; index positions n1, n2)'''
                C1 = lambda t: t[n1] > 0  # noqa: E731
                C2 = lambda t: t[n2] > 0  # noqa: E731
                inner = f'{a1} * {G(s1 + s2)}'
                if has(C1):
                    inner += f' - {c([t[n1] for t in ts])} * {G(-s1 + s2, C1)}'
                e = f'{a2} * ({inner})'
                if has(C2):
                    inner2 = f'{a1} * {G(s1 - s2, C2)}'
                    if has(lambda t: C1(t) and C2(t)):
                        inner2 += f' - {c([t[n1] for t in ts])} * {G(-s1 - s2, lambda t: C1(t) and C2(t))}'
                    e += f' - {c([t[n2] for t in ts])} * ({inner2})'
                return e
            # ii: g(i+2, j) = g(i+1, j+1) + rj g(i+1, j); jj: g(i, j+2) = g(i+1, j+1) - rj g(i, j+1)
            out['ii'] = same('ai2', 1, rj, '+', 0, 1 + sj)
            out['jj'] = same('aj2', sj, rj, '-', 1, 1 + sj)
            out['ij'] = mixed('ai2', 1, 0, 'aj2', sj, 1)
            out['kk'] = same('ak2', sk, rl, '+', 2, sk + sl)
            out['ll'] = same('al2', sl, rl, '-', 3, sk + sl)
            out['kl'] = mixed('ak2', sk, 2, 'al2', sl, 3)
        else:
            # i and k from the g arrays, j and l by the HRR: g(i, j+1) = g(i+1, j) - rj g(i, j), likewise for l
            def fi(s, cond=None):
                e = f'ai2 * {G(1 + s, cond)}'
                if has(lambda t: I(t) and (cond is None or cond(t))):
                    e += f' - {c([t[0] for t in ts])} * {G(-1 + s, lambda t: I(t) and (cond is None or cond(t)))}'
                return e

            def fj(s, cond=None):
                e = f'aj2 * ({G(1 + s, cond)} - {rj} * {G(s, cond)})'
                if has(lambda t: J(t) and (cond is None or cond(t))):
                    e += f' - {c([t[1] for t in ts])} * {G(-sj + s, lambda t: J(t) and (cond is None or cond(t)))}'
                return e

            def fi_l(s):
                '''fi of g(.., l+1) = g(.., k+1, l) - rl g(.., k, l)'''
                e = f'ai2 * ({G(1 + s)} - {rl} * {G(1)})'
                if has(I):
                    e += f' - {c([t[0] for t in ts])} * ({G(-1 + s, I)} - {rl} * {G(-1, I)})'
                return e

            def fj_l(s):
                e = f'aj2 * ({G(1 + s)} - {rj} * {G(s)} - {rl} * ({G(1)} - {rj} * {G(0)}))'
                if has(J):
                    e += f' - {c([t[1] for t in ts])} * ({G(-sj + s, J)} - {rl} * {G(-sj, J)})'
                return e
            for name, f in (('ik', fi), ('jk', fj)):
                e = f'ak2 * ({f(sk)})'
                if has(K):
                    e += f' - {c([t[2] for t in ts])} * ({f(-sk, K)})'
                out[name] = e
            for name, f, f0 in (('il', fi_l, fi), ('jl', fj_l, fj)):
                e = f'al2 * ({f(sk)})'
                if has(L):
                    e += f' - {c([t[3] for t in ts])} * ({f0(-sl, L)})'
                out[name] = e
        return out


def tuple_end(code, d, ts, I):
    '''Apply the factors of direction d at the lanes' tuples to S and T0-T3 (pass d, with the pair d, e)'''
    e = (d + 1) % 3
    out = [f'{I}Ai = {code.tab(d, ts, 0)};', f'{I}Aj = {code.tab(d, ts, 1)};',
           f'{I}Ak = {code.tab(d, ts, 2)};', f'{I}Al = {code.tab(d, ts, 3)};']
    if DROP_L:
        out += [f'{I}T3 = -(T0 + T1 + T2);']
    for grp, expr in code.second_factors(d, ts).items():
        out += [f'{I}{var(grp, d, d)} += ({expr}) * S;']
    # T0..T3: the sums of fi, fj, fk, fl of direction e
    if code.kind == 12:
        pairs = (('ij', d, e, 'Ai', 1), ('ij', e, d, 'Aj', 0), ('jj', d, e, 'Aj', 1), ('ii', d, e, 'Ai', 0),
                 ('kl', d, e, 'Ak', 3), ('kl', e, d, 'Al', 2), ('ll', d, e, 'Al', 3), ('kk', d, e, 'Ak', 2))
    else:
        pairs = (('ik', d, e, 'Ai', 2), ('ik', e, d, 'Ak', 0), ('jk', d, e, 'Aj', 2), ('jk', e, d, 'Ak', 1),
                 ('il', d, e, 'Ai', 3), ('il', e, d, 'Al', 0), ('jl', d, e, 'Aj', 3), ('jl', e, d, 'Al', 1))
    for grp, a, b, A, t in pairs:
        out += [f'{I}{var(grp, a, b)} += {A} * T{t};']
    return out


def pass_code(code, d, lanes, comps, k, I):
    '''Pass d of one warp group: lanes are k lists of (tuple, n) positions of equal length'''
    e, f = (d + 1) % 3, (d + 2) % 3
    nf = len(comps)
    nsq = code.nsq
    P = len(lanes[0])
    out = []
    if P == 0:
        return out
    if k > 1:
        # once per pass, so that a pass's selected addresses are computed in that pass
        out += [f'{I}asm volatile("mov.u32 %0, %%tid.y;" : "=r"(tid_y));', f'{I}h = tid_y % {k};']
    last = [None] * k
    fresh = True
    for p in range(P):
        pos = [lane[p] for lane in lanes]
        ns = [n for t, n in pos]
        cs = [comps[n] if n < nf else comps[0] for n in ns]
        last = [t if t is not None else last[h] for h, (t, n) in enumerate(pos)]
        op = '=' if fresh else '+='
        out += [f'{I}w = dd_cache[{sel([n * nsq for n in ns])}] * {code.g(f, [code.addr(tuple(x[f] for x in c)) for c in cs])};',
                f'{I}S {op} w * {code.g(e, [code.addr(tuple(x[e] for x in c)) for c in cs])};']
        te = [tuple(x[e] for x in c) for c in cs]
        for b in range(3 if DROP_L else 4):
            out += [f'{I}T{b} {op} {code.tab(e, te, b)} * w;']
        ends = p + 1 == P or any(lane[p + 1][0] is not None and lane[p + 1][0] != last[h]
                                 for h, lane in enumerate(lanes))
        fresh = ends
        if ends:
            out += tuple_end(code, d, [t if t is not None else (0, 0, 0, 0) for t in last], I)
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
BRA_SUMS
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
TASK_SUMS
        int do_j = jk.j_factor != 0.;
        int do_k = jk.k_factor != 0.;'''


def atomics(kind, scope, I):
    '''The atomicAdd of the sums of scope (bra: per bra pair, task: per task), as the generic kernels lay them out;
    the same-atom blocks (ii, jj, kk, ll) take half their diagonal and only their lower triangle'''
    atoms = {'i': 'ia', 'j': 'ja', 'k': 'ka', 'l': 'la'}
    out = []
    for grp, sc, n in GROUPS[kind]:
        if sc != scope:
            continue
        A, B = atoms[grp[0]], atoms[grp[1]]
        for a in range(3):
            for b in range(3):
                if grp[0] == grp[1]:
                    if a < b:
                        continue
                    v = var(grp, b, a) + ('*.5' if a == b else '')
                else:
                    v = var(grp, a, b)
                out += [f'{I}atomicAdd(ejk + ({A}*natm+{B})*9 + {3 * a + b}, {v});']
    return out


def kernel(kind, li, lj, lk, ll, nsq, gs):
    name = f'rys_ejk_ip2_type{kind}_gen_{li}{lj}{lk}{ll}'
    code = Code(kind, li, lj, lk, ll, nsq)
    rec, sj, sk, sl, G = layout(kind, li, lj, lk, ll)
    NT = code.NT
    nfi, nfj, nfk = len(cart(li)), len(cart(lj)), len(cart(lk))
    k = max(1, 32 // nsq)   # gout_ids per warp
    comps, runs = passes(li, lj, lk, ll, gs, k)
    nf = len(comps)
    nfz = nf + (k > 1)      # a zero slot for the padding of the lanes
    assert nfz * nsq <= DD_CACHE_MAX, (name, nfz * nsq)
    nroots_sr = 2 * ((li + lj + lk + ll + 2) // 2 + 1)
    smem = nsq * (6 + 3 * G + 12 * NT + 2 * nroots_sr)
    assert smem * 8 + 64 * 8 + 128 <= 65536, (name, 'shared memory', smem * 8)
    I3, I4, I5 = ' ' * 12, ' ' * 16, ' ' * 20
    L = [f'// type{kind} ({"spdfghi"[li]}{"spdfghi"[lj]}|{"spdfghi"[lk]}{"spdfghi"[ll]}): {nf} outputs, {NT} tuples '
         f'per direction, {gs} threads x {nsq} quartets per block, g_size {G}', '__global__ static',
         f'__launch_bounds__({nsq * gs}, 1)', f'void {name}(JKENERGY_KERNEL_ARGS)', '{',
         '    JKENERGY_KERNEL_SETUP();', f'    constexpr int nsq_per_block = {nsq};', f'    constexpr int gout_stride = {gs};',
         '    int t_id = gout_id * nsq_per_block + sq_id;', '    int threads = nsq_per_block * gout_stride;',
         '    int nbas = envs.nbas;', '    int *bas = envs.bas;', '    double *env = envs.env;',
         '    int iprim = bounds.iprim;', '    int jprim = bounds.jprim;', '    int kprim = bounds.kprim;',
         '    int lprim = bounds.lprim;', '    int nroots = bounds.nroots;', '    double *rlrk = shared_memory + sq_id;',
         '    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;',
         '    // the g arrays of the three directions, then the first-derivative factors of every tuple',
         '    double *gx = shared_memory + nsq_per_block * 6 + sq_id;',
         f'    double *rw = shared_memory + nsq_per_block * {6 + 3 * G + 12 * NT} + sq_id;',
         f'    double *cicj_cache = shared_memory + nsq_per_block * ({6 + 3 * G + 12 * NT}+nroots*2);', '',
         '    uint32_t *bas_kl_idx = pool + worker_id * QUEUE_DEPTH;',
         f'    double *dd_cache = dd_pool + worker_id * {nfz * nsq} + sq_id;']
    bra = ''.join(f'    double {v} = 0;\n' for v in group_vars(kind, 'bra')).rstrip('\n')
    task = ''.join(f'        double {v} = 0;\n' for v in group_vars(kind, 'task')).rstrip('\n')
    L += PROLOGUE.replace('NSQ', str(nsq)).replace('BRA_SUMS', bra).replace('TASK_SUMS', task).splitlines()
    # density products of the task's outputs, n = i + nfi*(j + nfj*(k + nfk*l)), as the generic kernels
    for n_dm in (1, 2):
        L += [f'        {"if (jk.n_dm == 1) {" if n_dm == 1 else "} else {"}']
        if n_dm == 2:
            L += ['            double *dmb = dm + nao * nao;']
        L += [f'            for (int n = gout_id; n < {nf}; n+={gs}) {{',
              f'                int i = n % {nfi};', f'                int j = n / {nfi} % {nfj};',
              f'                int k = n / {nfi * nfj} % {nfk};', f'                int l = n / {nfi * nfj * nfk};',
              '                int _i = i + i0;', '                int _j = j + j0;', '                int _k = k + k0;',
              '                int _l = l + l0;', '                int _jl = _j*nao+_l;', '                int _jk = _j*nao+_k;',
              '                int _il = _i*nao+_l;', '                int _ik = _i*nao+_k;', '                int _ji = _j*nao+_i;',
              '                int _lk = _l*nao+_k;', '                double dd = 0;']
        if n_dm == 1:
            L += ['                if (do_k) {',
                  '                    dd += jk.k_factor * (dm[_jk] * dm[_il] + dm[_jl] * dm[_ik]);', '                }',
                  '                if (do_j) {', '                    dd += jk.j_factor * dm[_ji] * dm[_lk];', '                }']
        else:
            L += ['                if (do_k) {', '                    dd += dm [_jk] * dm [_il] + dm [_jl] * dm [_ik];',
                  '                    dd += dmb[_jk] * dmb[_il] + dmb[_jl] * dmb[_ik];',
                  '                    dd *= jk.k_factor;', '                }', '                if (do_j) {',
                  '                    dd += jk.j_factor * (dm[_ji] + dmb[_ji]) * (dm[_lk] + dmb[_lk]);', '                }']
        L += [f'                dd_cache[n*{nsq}] = fac_sym * dd;', '            }']
    L += ['        }']
    if k > 1:
        L += ['        if (gout_id == 0) {', f'            dd_cache[{nf * nsq}] = 0;', '        }']
    L += PRIM_LOOPS.replace('NSQ', str(nsq)).replace('GSIZE', str(G)).replace('GS', str(gs)).splitlines()
    # root loop: TRR and HRR one direction per thread, the factor table, then the three passes
    lij, lkl = rec[0] + rec[1], rec[2] + rec[3]
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
    body += [f'{I4}        {s}' for s in recursions(*rec, nsq)]
    body += [f'{I4}    }}', f'{I4}    __syncthreads();']
    # the first-derivative factors fi, fj, fk, fl of every tuple of every direction (type3: j and l by the HRR)
    fj = (f'aj2 * _g[{sj * nsq}]' if kind == 12 else f'aj2 * (_g[{nsq}] - rjri[d] * _g[0])')
    fl = (f'al2 * _g[{sl * nsq}]' if kind == 12 else f'al2 * (_g[{sk * nsq}] - rlrk[d*{nsq}] * _g[0])')
    body += [f'{I4}    for (int e = gout_id; e < {3 * NT}; e += {gs}) {{', f'{I4}        int d = e / {NT};',
             f'{I4}        int t = e - d * {NT};', f'{I4}        int ti = t % {li + 1};',
             f'{I4}        int tj = t / {li + 1} % {lj + 1};', f'{I4}        int tk = t / {(li + 1) * (lj + 1)} % {lk + 1};',
             f'{I4}        int tl = t / {(li + 1) * (lj + 1) * (lk + 1)};',
             f'{I4}        double *_g = gx + (d * {G} + ti + tj * {sj} + tk * {sk} + tl * {sl}) * {nsq};',
             f'{I4}        double *_t = gx + ({3 * G} + e * 4) * {nsq};',
             f'{I4}        _t[0] = ai2 * _g[{nsq}] - (ti ? ti * _g[-{nsq}] : 0.);',
             f'{I4}        _t[{nsq}] = {fj} - (tj ? tj * _g[-{sj * nsq}] : 0.);',
             f'{I4}        _t[{2 * nsq}] = ak2 * _g[{sk * nsq}] - (tk ? tk * _g[-{sk * nsq}] : 0.);',
             f'{I4}        _t[{3 * nsq}] = {fl} - (tl ? tl * _g[-{sl * nsq}] : 0.);', f'{I4}    }}',
             f'{I4}    __syncthreads();',
             # volatile: otherwise the compiler keeps every value it loaded for a later run in registers
             f'{I4}    volatile double *vgx = gx;', f'{I4}    double w, S, T0, T1, T2, T3, Ai, Aj, Ak, Al;']
    if k > 1:
        body += [f'{I4}    unsigned tid_y;', f'{I4}    int h;']
    body += [f'{I4}    switch (gout_id / {k}) {{']
    for w in range(gs // k):
        body += [f'{I4}    case {w}:']
        for d in range(3):
            body += pass_code(code, d, runs[d][w], comps, k, I5)
        body += [f'{I5}break;']
    body += [f'{I4}    }}', f'{I4}}}']
    L += body + [f'{I3}}}', '        }']
    L += ['        if (task_id >= ntasks) {', '            break;', '        }',
          '        int ia = bas[ish*BAS_SLOTS+ATOM_OF];', '        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];',
          '        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];', '        int la = bas[lsh*BAS_SLOTS+ATOM_OF];',
          '        int natm = envs.natm;', '        double *ejk = jk.ejk;']
    L += atomics(kind, 'task', '        ')
    L += ['    }']
    if group_vars(kind, 'bra'):
        L += ['    int ia = bas[ish*BAS_SLOTS+ATOM_OF];', '    int ja = bas[jsh*BAS_SLOTS+ATOM_OF];',
              '    int natm = envs.natm;', '    double *ejk = jk.ejk;']
        L += atomics(kind, 'bra', '    ')
    L += ['}', '}', '']
    return name, G, NT, nf, '\n'.join(L), smem


HEADER = '''// Generated by bench/gpu/gen_rys_ejk_ip2.py of the my-pyscf workspace: multi-thread Rys kernels for the
// Hessian's two-electron second derivatives (RYS_per_atom_jk_ip2_type12, _type3) that sum the outputs per
// Cartesian tuple before applying the derivative factors. Do not edit by hand.
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
    for ijkl, kname, nsq, gs, G, NT in cases:
        disp += [f'    case {ijkl}: {{', f'        dim3 threads({nsq}, {gs});',
                 f'        int buflen = {nsq} * ({6 + 3 * G + 12 * NT} + nroots*2) + ijprim;',
                 '        if (buflen*sizeof(double) > 48000) {',
                 f'            cudaFuncSetAttribute({kname}, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));',
                 '        }', f'        LAUNCH_JKENERGY_KERNEL({kname});', '        return 1; }']
    return disp + ['    }', '    return 0;', '}', '']


ARGS = ('RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds, float *q_cond_ij, float *q_cond_kl, '
        'float dm_penalty, float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, uint32_t *pool, double *dd_pool, '
        'int *head, int workers')
CALL = 'envs, jk, bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head, workers'


def main():
    global DROP_L
    args = sys.argv[1:]
    if '--drop-l' in args:
        args.remove('--drop-l')
        DROP_L = True
    n_files = 1
    if args[1] == '--files':
        n_files = int(args[2])
        args = args[:1] + args[3:]
    out, specs = args[0], args[1:]
    built = []
    for spec in specs:
        kind, cls, *shape = spec.split(':')
        kind = int(kind)
        assert kind in (12, 3), spec
        li, lj, lk, ll = (int(c) for c in cls)
        nsq, gs = (int(x) for x in (shape[0] if shape else '16x16').split('x'))
        assert nsq in (8, 16, 32, 64) and nsq * gs <= 512 and gs % max(1, 32 // nsq) == 0, spec
        name, G, NT, nf, text, smem = kernel(kind, li, lj, lk, ll, nsq, gs)
        built.append((text, kind, (li * 125 + lj * 25 + lk * 5 + ll, name, nsq, gs, G, NT)))
        print(f'{name}: {nsq}x{gs}, {nf} outputs, {NT} tuples, shared memory up to {smem * 8} bytes',
              file=sys.stderr)
    # several translation units, balanced by size, so that the library compiles in parallel
    parts = [[] for _ in range(n_files)]
    for item in sorted(built, key=lambda b: -len(b[0])):
        min(parts, key=lambda q: sum(len(t) for t, kd, c in q)).append(item)
    base = out[:-3]
    for f, part in enumerate(parts):
        text = HEADER + '\n'.join(t for t, kd, c in part) + '\n'
        for kind in (12, 3):
            cases = [c for t, kd, c in part if kd == kind]
            fn = f'rys_ejk_ip2_type{kind}_generated'
            if n_files == 1:
                text += '\n'.join(dispatcher(f'int {fn}', cases))
            elif f == 0:
                text += ''.join(f'extern int {fn}_{g}({ARGS});\n' for g in range(1, n_files))
                text += '\n'.join(dispatcher(f'static int {fn}_0', cases))
                text += f'int {fn}({ARGS})\n{{\n    return ' + ' ||\n        '.join(
                    f'{fn}_{g}({CALL})' for g in range(n_files)) + ';\n}\n'
            else:
                text += '\n'.join(dispatcher(f'int {fn}_{f}', cases))
        open(out if f == 0 else f'{base}_{f}.cu', 'w').write(text)


if __name__ == '__main__':
    main()
