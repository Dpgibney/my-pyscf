"""The VV10 kernels of one PySCF tree on fixed inputs, for comparing builds (It24).
  --make IN.npz   caffeine/def2-SVP, the minao guess density on an NLC grid of --level: rho, coords, weights
  IN.npz OUT.npz  numint._vv10nlc with the outer grid the inner one (the SCF's call: VXC_vv10nlc_sym in the
                  modified tree), the same with copies of the arrays (VXC_vv10nlc), and grad.rks._vv10nlc_grad
                  with the first --nout points as the outer grid (VXC_vv10nlc_grad); each timed, best of --reps
  --compare A.npz B.npz   the largest differences, relative to the largest values, and the NLC energy's
Run with PYTHONPATH set to the tree, e.g. PYTHONPATH=pyscf-base .venv/bin/python bench/diag/vv10_check.py IN OUT
Usage: vv10_check.py --make IN.npz [--level 1] | IN.npz OUT.npz [--nout 20000] [--reps 3] | --compare A.npz B.npz"""
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
NLC_PARS = (6.0, 0.01)  # wB97M-V's VV10 b and C


def make(path, level):
    from pyscf import gto, dft
    from pyscf.dft import numint
    from molecules import CAFFEINE
    mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    dm = dft.RKS(mol).get_init_guess(key='minao')
    ni = numint.NumInt()
    rho = numpy.hstack([ni.eval_rho(mol, ao, dm, xctype='GGA')
                        for ao, mask, weight, coords in ni.block_loop(mol, grids, mol.nao, 1)])
    numpy.savez(path, rho=rho, coords=grids.coords, weights=grids.weights)
    print(f'{path}: {grids.weights.size} points (level {level})')


def run(inp, out, nout, reps):
    from pyscf.dft import numint
    from pyscf.grad import rks as rks_grad
    d = numpy.load(inp)
    rho, coords, weights = d['rho'], d['coords'], d['weights']
    res, times = {}, {}

    def best(name, fn):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            v = fn()
            ts.append(time.perf_counter() - t0)
        times[name] = min(ts)
        return v
    exc, vxc = best('same_grid', lambda: numint._vv10nlc(rho, coords, rho, weights, coords, NLC_PARS))
    res.update(same_exc=exc, same_vxc=vxc)
    exc, vxc = best('two_grids', lambda: numint._vv10nlc(rho, coords, rho.copy(), weights, coords.copy(), NLC_PARS))
    res.update(two_exc=exc, two_vxc=vxc)
    g, beta = best('grad', lambda: rks_grad._vv10nlc_grad(rho[:, :nout], coords[:nout], rho, weights, coords,
                                                          NLC_PARS))
    res.update(grad=g)
    res['e_same'] = numpy.dot(rho[0] * weights, res['same_exc'])
    res['e_two'] = numpy.dot(rho[0] * weights, res['two_exc'])
    numpy.savez(out, **res)
    print(f'{os.path.dirname(numint.__file__)}: {weights.size} points, ' +
          ', '.join(f'{k} {v:.3f} s' for k, v in times.items()) + f'; E_nlc {res["e_same"]:.12f} / {res["e_two"]:.12f}')


def compare(a, b):
    A, B = numpy.load(a), numpy.load(b)
    for k in A.files:
        x, y = A[k], B[k]
        diff = numpy.abs(x - y).max()
        print(f'  {k:9s} max |diff| {diff:.2e}, relative to max |value| {diff / max(numpy.abs(x).max(), 1e-300):.2e}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == '--make':
        make(a[1], int(a[a.index('--level') + 1]) if '--level' in a else 1)
    elif a[0] == '--compare':
        compare(a[1], a[2])
    else:
        run(a[0], a[1], int(a[a.index('--nout') + 1]) if '--nout' in a else 20000,
            int(a[a.index('--reps') + 1]) if '--reps' in a else 3)
