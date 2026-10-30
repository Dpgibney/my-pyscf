"""Thread pools: CPU that numpy's and scipy's bundled OpenBLAS and PySCF's OpenMP threads burn after a call,
and what that costs a short PySCF OpenMP call made next.

Also times scipy's eigh and numpy's GEMM at three sizes, which the pool settings slow down in
different ways. Run it in each configuration, e.g.
    bench/diag/run_in.py N=pyscf:OPENBLAS_THREAD_TIMEOUT=28 10 bench/diag/blas_pools.py  (OpenBLAS default)
    bench/diag/run_in.py T=pyscf 10 bench/diag/blas_pools.py  (the harness's pinned 20)
"""
import os
import resource
import sys
import time

import numpy
import scipy.linalg

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cases  # noqa: E402
from pyscf import gto, lib  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def burn(call, pause=.5, reps=3):
    '''CPU seconds all threads use during a pause of the main thread right after call().'''
    out = []
    for _ in range(reps):
        call()
        c0 = cpu()
        time.sleep(pause)
        out.append(cpu() - c0)
    return numpy.median(out)


def timed(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def probe_after(call, probe, reps=15):
    '''Median time of probe() right after call(), and after a 0.4 s pause.'''
    hot, cold = [], []
    for _ in range(reps):
        call()
        hot.append(timed(probe))
        time.sleep(.4)
        cold.append(timed(probe))
    return numpy.median(hot), numpy.median(cold)


def call_times(call, reps=15):
    '''Median time of call() run back to back, and after a 0.4 s pause.'''
    call()
    back = [timed(call) for _ in range(reps)]
    after_pause = []
    for _ in range(reps):
        time.sleep(.4)
        after_pause.append(timed(call))
    return numpy.median(back), numpy.median(after_pause)


def main():
    env = {k: v for k, v in os.environ.items() if k.startswith(('OPENBLAS_', 'OMP_', 'GOMP_', 'MALLOC_'))}
    print('env:', ' '.join(f'{k}={v}' for k, v in sorted(env.items())))
    with open('/proc/self/maps') as f:
        libs = sorted({line.split()[-1] for line in f if 'blas' in line.lower()})
    mol = gto.M(atom=cases.CAFFEINE, basis='def2-svp', verbose=0)
    nao = mol.nao
    rng = numpy.random.default_rng(1)
    a = rng.random((nao, nao))
    f = a + a.T
    s = mol.intor('int1e_ovlp')
    dm = f @ f.T
    v = rng.random(128*1000)
    ao_loc = mol.ao_loc_nr()
    from pyscf.scf import _vhf
    eri = mol.intor('int2e', aosym='s8')
    def jk(d):
        return _vhf.incore(eri, d)
    print('BLAS libraries:', ', '.join(os.path.basename(p) for p in libs if p.startswith('/')))
    calls = {
        'numpy a@a': lambda: a @ a,
        'numpy ddot 128k': lambda: numpy.dot(v, v),
        'scipy eigh(F,S)': lambda: scipy.linalg.eigh(f, s),
        'lib.dot (OpenMP)': lambda: lib.dot(a, a),
        'eigh, then a@a': lambda: (scipy.linalg.eigh(f, s), a @ a),
    }
    probes = {
        'get_overlap_cond': lambda: mol.get_overlap_cond(),
        'condense': lambda: lib.condense('NP_absmax', dm, ao_loc),
        'J+K in core': lambda: jk(dm),
    }
    print(f'nao={nao}; idle CPU (s) in 0.5 s after the call, then the call itself (ms) back to back / after a pause')
    for name, call in calls.items():
        back, cold = call_times(call)
        print(f'  {name:18s} idle CPU {burn(call):5.2f} s   call {back*1e3:7.2f} / {cold*1e3:7.2f} ms')
    print('probe time (ms) right after the call / after a pause')
    for pname, probe in probes.items():
        for name, call in calls.items():
            hot, cold = probe_after(call, probe)
            print(f'  {pname:16s} after {name:18s} {hot*1e3:7.3f} / {cold*1e3:7.3f} ms')
    print('scipy eigh(F,S) and numpy GEMM, back to back (median ms)')
    for n in (246, 520, 872):
        a = rng.random((n, n))
        b = rng.random((n, n))
        f, s = a + a.T, b @ b.T / n + numpy.eye(n)
        t_eigh = call_times(lambda: scipy.linalg.eigh(f, s), reps=9)[0]
        t_gemm = call_times(lambda: a @ a, reps=9)[0]
        print(f'  n={n:4d} eigh {t_eigh*1e3:6.1f} ms  a@a {t_gemm*1e3:6.2f} ms')


if __name__ == '__main__':
    main()
