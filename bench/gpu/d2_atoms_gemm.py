"""Prototype of part 14's change B (node1): the per-atom terms of the Hessian's _get_vxc_deriv2 for one grid block
as one GEMM over atoms (t = each AO's terms of dR_rho1, an atom's dR_rho1 = the sum of t over its AOs, its terms
= 1/2 (fxc dR_rho1) . t), against a plain-CuPy transcription of the per-atom loop it replaced, on random inputs shaped
like a caffeine/def2-SVP block (170 AOs, 24 atoms, 4096 points) and an n-C20H42 one. Prints the agreement and the time
of each piece per block. Usage: d2_atoms_gemm.py (source scripts/node1-gpu-env.sh perf first)."""
import time
import numpy
import cupy

_k = {}
def _terms_kernel(ncomp):
    name = f'dR_rho1_terms_{ncomp}'
    if name not in _k:
        body = '''
        for (int x = 0; x < 3; ++x) t[x*sx] = 2. * a[x] * d0;'''
        if ncomp > 1:
            body = '''
        double d[3] = {dm[N+e], dm[2*N+e], dm[3*N+e]};
        double h0 = ao[4*N+e], h1 = ao[5*N+e], h2 = ao[6*N+e];
        double h3 = ao[7*N+e], h4 = ao[8*N+e], h5 = ao[9*N+e];
        double h[3][3] = {{h0, h1, h2}, {h1, h3, h4}, {h2, h4, h5}};
        for (int x = 0; x < 3; ++x) {
            double *tx = t + x*sx;
            tx[0] = 2. * a[x] * d0;
            for (int c = 0; c < 3; ++c) tx[(c+1)*ng] = 2. * (h[x][c] * d0 + a[x] * d[c]);''' + ('''
            tx[4*ng] = h[x][0] * d[0] + h[x][1] * d[1] + h[x][2] * d[2];''' if ncomp == 5 else '') + '''
        }'''
        code = r'''
extern "C" __global__
void ''' + name + r'''(double *T, const double *ao, const double *dm, int n, int ng)
{
    const size_t N = (size_t)n * ng;
    const size_t sx = N * ''' + str(ncomp) + r''';
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < N;
         e += (size_t)gridDim.x * blockDim.x) {
        size_t j = e / ng;
        size_t g = e - j * ng;
        double d0 = dm[e];
        double a[3] = {ao[N+e], ao[2*N+e], ao[3*N+e]};
        double *t = T + j * ''' + str(ncomp) + r''' * ng + g;''' + body + r'''
    }
}'''
        _k[name] = cupy.RawKernel(code, name)
    return _k[name]

def dR_rho1_terms(ao, dm, ncomp):
    n, ng = ao.shape[1:]
    t = cupy.empty((3, n, ncomp, ng))
    _terms_kernel(ncomp)((min(4096, (n*ng+255)//256),), (256,), (t, ao, dm, numpy.int32(n), numpy.int32(ng)))
    return t

_sum_code = r'''
extern "C" __global__
void atom_row_sums(double *R, const double *T, const int *offsets, int n, int m, long long ldr)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y;
    int x = blockIdx.z;
    if (k >= m) return;
    const double *t = T + (size_t)x * n * m + k;
    double s = 0.;
    for (int j = offsets[a]; j < offsets[a+1]; ++j) s += t[(size_t)j * m];
    R[((size_t)a * 3 + x) * ldr + k] = s;
}'''
_sum_kernel = cupy.RawKernel(_sum_code, 'atom_row_sums')

def atom_sums(t, offsets):
    _, n, ncomp, ng = t.shape
    m = ncomp * ng
    na = len(offsets) - 1
    r = cupy.empty((na, 3, ncomp, ng))
    _sum_kernel(((m+255)//256, na, 3), (256,), (r, t, cupy.asarray(offsets, dtype=numpy.int32),
                                               numpy.int32(n), numpy.int32(m), numpy.int64(m)))
    return r

def contract_rho1_fxc(rho1, fxc):
    nvar, ng = fxc.shape[-2:]
    shape = rho1.shape
    return (rho1.reshape(-1,1,nvar,ng) * fxc).sum(axis=2).reshape(shape)

XX, XY, XZ, YY, YZ, ZZ = 4, 5, 6, 7, 8, 9
H = [[XX, XY, XZ], [XY, YY, YZ], [XZ, YZ, ZZ]]

def ref_atom(ao, dm, p0, p1, wf, ncomp):
    """stage A per-atom code (as in _make_dR_rho1 and the loop), in plain cupy"""
    a1 = ao[1:4, p0:p1]
    if ncomp == 1:
        rho1 = 2 * (a1 * dm[0, p0:p1]).sum(axis=1)
        wv = wf * rho1
        aow = dm[0][None] * wv[:, None]
        return (aow[:, None] * ao[None, 1:4]).sum(axis=-1)   # [i, x, j]
    rho1 = cupy.zeros((3, ncomp, ao.shape[2]))
    for x in range(3):
        rho1[x, 0] = (ao[1+x, p0:p1] * dm[0, p0:p1]).sum(axis=0)
        for c in range(3):
            rho1[x, c+1] = (ao[H[x][c], p0:p1] * dm[0, p0:p1] + ao[1+x, p0:p1] * dm[1+c, p0:p1]).sum(axis=0)
        if ncomp == 5:
            rho1[x, 4] = .5 * sum((ao[H[x][k], p0:p1] * dm[1+k, p0:p1]).sum(axis=0) for k in range(3))
    rho1 *= 2
    wv = contract_rho1_fxc(rho1, wf)
    wv[:, 0] *= .5
    if ncomp == 5:
        wv[:, 4] *= .5
    v = cupy.zeros((3, 3, ao.shape[1]))
    for i in range(3):
        for x in range(3):
            aow = wv[i, 0] * ao[1+x] + sum(wv[i, 1+k] * ao[H[x][k]] for k in range(3))
            v[i, x] = (aow * dm[0]).sum(axis=-1)
            v[i, x] += (sum(dm[c] * wv[i, c] for c in range(4)) * ao[1+x]).sum(axis=-1)
            if ncomp == 5:
                v[i, x] += sum((dm[1+k] * wv[i, 4] * ao[H[x][k]]).sum(axis=-1) for k in range(3))
    return v

def stage_b(ao, dm, offsets, wf, ncomp):
    t = dR_rho1_terms(ao, dm, ncomp)
    r = atom_sums(t, offsets)
    na = len(offsets) - 1
    n, ng = ao.shape[1:]
    if ncomp == 1:
        w = r.reshape(na*3, ng) * (.5 * wf)
    else:
        w = contract_rho1_fxc(r.reshape(na*3, ncomp, ng), .5 * wf)
    v = w.reshape(na*3, ncomp*ng).dot(t.reshape(3*n, ncomp*ng).T)
    return v.reshape(na, 3, 3, n), t

rng = numpy.random.default_rng(1)
for ncomp, n, na in ((1, 170, 24), (4, 170, 24), (5, 170, 24), (4, 172, 33)):
    ng = 4096
    nd = 4 if ncomp == 1 else 10
    ao = cupy.asarray(rng.standard_normal((nd, n, ng)))
    dm = cupy.asarray(rng.standard_normal((1 if ncomp == 1 else 4, n, ng)))
    sizes = rng.multinomial(n - na, numpy.ones(na) / na) + 1
    offsets = numpy.concatenate(([0], numpy.cumsum(sizes)))
    f = rng.standard_normal((ncomp, ncomp, ng))
    wf = cupy.asarray(f + f.transpose(1, 0, 2)) if ncomp > 1 else cupy.asarray(rng.standard_normal(ng))
    vb, t = stage_b(ao, dm, offsets, wf, ncomp)
    err = 0.
    for a in range(na):
        ref = ref_atom(ao, dm, offsets[a], offsets[a+1], wf, ncomp)
        err = max(err, float(abs(vb[a] - ref).max() / abs(ref).max()))
    # timings: the pieces of stage B
    def tm(fn, rep=10):
        fn(); cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        cupy.cuda.Device().synchronize()
        return (time.perf_counter() - t0) / rep * 1e3
    t_terms = tm(lambda: dR_rho1_terms(ao, dm, ncomp))
    t_sums = tm(lambda: atom_sums(t, offsets))
    r = atom_sums(t, offsets)
    if ncomp == 1:
        t_fxc = tm(lambda: r.reshape(na*3, ng) * wf)
    else:
        t_fxc = tm(lambda: contract_rho1_fxc(r.reshape(na*3, ncomp, ng), wf))
    w = cupy.asarray(rng.standard_normal((na*3, ncomp*ng)))
    t_gemm = tm(lambda: w.dot(t.reshape(3*n, ncomp*ng).T))
    gflop = 2 * na*3 * 3*n * ncomp*ng / 1e9
    t_all = tm(lambda: stage_b(ao, dm, offsets, wf, ncomp))
    print(f'ncomp {ncomp} n {n} atoms {na}: rel err {err:.1e}; per block terms {t_terms:.2f} ms, atom sums {t_sums:.2f} ms, '
          f'fxc {t_fxc:.2f} ms, GEMM {t_gemm:.2f} ms ({gflop/t_gemm*1e3:.0f} GFLOP/s), all {t_all:.2f} ms')
