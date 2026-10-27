"""Run one case in this process and print one JSON record (prefixed by MARK) to stdout.

Usage: worker.py <case> [--tight]
Launched by run_bench.py/validate.py with the environment that selects the PySCF tree and settings.
"""
import ctypes
import cProfile
import json
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases  # noqa: E402

MARK = '@@RESULT@@'

# (category, file suffix, function name or '*'); first match wins. Times are cProfile tottime, so a C
# routine called through ctypes is charged to the Python wrapper that calls it.
CATEGORY_RULES = [
    ('jk', '_vhf.py', '*'), ('jk', 'moleintor.py', 'getints4c'), ('jk', 'moleintor.py', 'getints3c'),
    ('blas', 'numpy_helper.py', '_dgemm'), ('blas', 'numpy_helper.py', '_zgemm'),
    ('ao_eval', 'eval_gto.py', 'eval_gto'),
    ('xc_kernels', 'numint.py', '_scale_ao'), ('xc_kernels', 'numint.py', '_contract_rho'),
    ('xc_kernels', 'numint.py', '_scale_ao_sparse'), ('xc_kernels', 'numint.py', '_contract_rho_sparse'),
    ('xc_kernels', 'numint.py', '_dot_ao_ao_sparse'), ('xc_kernels', 'numint.py', '_dot_ao_dm_sparse'),
    ('xc_kernels', 'numint.py', '_tau_dot_sparse'), ('xc_kernels', 'numint.py', '_dot_ao_ao'),
    ('xc_kernels', 'numint.py', '_dot_ao_dm'),
    ('libxc', 'libxc.py', '_eval_xc'), ('xc_post', 'xc_deriv.py', '*'),
    ('grids', 'gen_grid.py', '*'), ('grids', 'radi.py', '*'),
    ('linalg', '_decomp.py', '*'), ('linalg', '_basic.py', '*'), ('linalg', 'linalg.py', '*'),
    ('ints1e', 'moleintor.py', '*'),
]


def categorize(stats):
    totals = {}
    for (fname, _, func), (_, _, tottime, _, _) in stats.stats.items():
        cat = 'other'
        for name, suffix, fn in CATEGORY_RULES:
            if fname.endswith(suffix) and (fn == '*' or fn == func):
                cat = name
                break
        if cat == 'other' and fname == '~' and ('einsum' in func or 'numpy' in func):
            cat = 'numpy'
        totals[cat] = totals.get(cat, 0.) + tottime
    return totals


def top_functions(stats, n=12):
    rows = sorted(stats.stats.items(), key=lambda kv: -kv[1][2])[:n]
    out = []
    for (fname, line, func), (_, ncalls, tottime, _, _) in rows:
        short = f'{"/".join(fname.split("/")[-2:])}:{line}({func})' if fname != '~' else func
        out.append([short, round(tottime, 4), ncalls])
    return out


def loaded_libs():
    keys = ('libopenblas', 'libcint', 'libxc', 'libdft', 'libgomp', 'libmkl', 'libblas')
    libs = set()
    with open('/proc/self/maps') as f:
        for line in f:
            path = line.split()[-1]
            if path.startswith('/') and any(k in os.path.basename(path) for k in keys):
                libs.add(os.path.realpath(path))
    return sorted(libs)


def blas_core(libs):
    for path in libs:
        if 'libopenblas' in os.path.basename(path):
            lib = ctypes.CDLL(path)
            for sym in ('openblas_get_corename', 'openblas_get_corename64_'):
                if hasattr(lib, sym):
                    fn = getattr(lib, sym)
                    fn.restype = ctypes.c_char_p
                    return fn().decode()
    return None


def warmup():
    '''Import the modules and load the libraries the cases use, so they are not charged to the timing.'''
    from pyscf import gto, dft, grad, tdscf, hessian, df  # noqa: F401
    import pyscf.grad.rks, pyscf.hessian.rks, pyscf.tdscf.rks  # noqa: F401,E401
    mol = gto.M(atom='O 0 0 0; H 0 0.76 -0.47; H 0 -0.76 -0.47', basis='6-31g', verbose=0)
    mf = dft.RKS(mol, xc='B3LYP')
    mf.chkfile = None
    mf.kernel()


def main():
    name = sys.argv[1]
    tight = '--tight' in sys.argv[2:]
    import pyscf
    from pyscf import lib
    warmup()
    setup, _ = cases.CASES[name]
    run = setup(tight)
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    values = run()
    prof.disable()
    wall = time.perf_counter() - t0
    stats = pstats.Stats(prof)
    libs = loaded_libs()
    record = {
        'case': name, 'tight': tight, 'wall': wall, 'values': values,
        'categories': categorize(stats), 'top': top_functions(stats),
        'pyscf_file': pyscf.__file__, 'pyscf_version': pyscf.__version__,
        'omp_threads': lib.num_threads(), 'libs': libs, 'blas_core': blas_core(libs),
        'max_rss_mb': __import__('resource').getrusage(0).ru_maxrss / 1024,
    }
    print(MARK + json.dumps(record))


if __name__ == '__main__':
    main()
