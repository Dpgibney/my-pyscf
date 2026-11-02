"""Where a GPU4PySCF Hessian, ground-state gradient or TDDFT gradient spends its time (node1). The SCF (and for tdgrad
the excited states) runs first, untimed; then the Hessian, or the gradient, runs once with timers around its stages,
each between device synchronizations. A stage's time includes the stages it calls; within a group (all J/K builds,
all XC responses) only the outermost call counts. The ground-state gradient (grad) is first run once untimed, since
its first call carries one-time start-up costs (in a GGA job, cuTENSOR's: NOTES part 18); that call's total is
printed as well, and with --first-stages its stages too. Source scripts/node1-gpu-env.sh perf|base first.
Usage: stage_times.py hess|grad|tdgrad [--xc b3lyp] [--basis def2-svp] [--mol caffeine] [--rpa] [--nstates 5]
                      [--state 1] [--first-stages] [--out FILE]"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft  # noqa: E402
from gpu4pyscf.dft import numint  # noqa: E402
from gpu4pyscf.scf import cphf, j_engine, jk  # noqa: E402
from gpu4pyscf.hessian import rhf as rhf_hess, rks as rks_hess  # noqa: E402
from gpu4pyscf.grad import rhf as rhf_grad, rks as rks_grad  # noqa: E402
from gpu4pyscf.grad import tdrhf as tdrhf_grad, tdrks as tdrks_grad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('what', choices=['hess', 'grad', 'tdgrad'])
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--rpa', action='store_true', help='TDDFT instead of TDA')
ap.add_argument('--nstates', type=int, default=5)
ap.add_argument('--state', type=int, default=1)
ap.add_argument('--first-stages', action='store_true', help='grad: time the first call\'s stages as well')
ap.add_argument('--out', default=None)
args = ap.parse_args()

times, calls, depth = defaultdict(float), defaultdict(int), defaultdict(int)


def timed(label, f, group=None):
    '''f with a synchronizing timer under label; calls nested inside another call of the same group are not counted'''
    group = group or label

    def wrapper(*a, **kw):
        if depth[group]:
            return f(*a, **kw)
        depth[group] += 1
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        try:
            return f(*a, **kw)
        finally:
            cupy.cuda.Device().synchronize()
            times[label] += time.perf_counter() - t0
            calls[label] += 1
            depth[group] -= 1
    return wrapper


def patch(obj, name, label, group=None):
    setattr(obj, name, timed(label, getattr(obj, name), group))


mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.kernel()
first = None
if args.what == 'grad':
    job = mf.nuc_grad_method()
    run = job.kernel
    if not args.first_stages:
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        job.kernel()
        cupy.cuda.Device().synchronize()
        first = time.perf_counter() - t0
    patch(type(job), '_hcore_energy', '1e terms (_hcore_energy)')
    patch(rhf_grad, '_grad_nuc_without_ecp', 'of it: kinetic and nuclear attraction (_grad_nuc_without_ecp)')
    patch(type(job), 'get_ovlp', 'of it: overlap derivatives (get_ovlp)')
    patch(type(job), 'make_rdm1e', 'energy-weighted density (make_rdm1e)')
    patch(rks_grad, 'get_exc', 'XC term (get_exc)')
    patch(type(job), 'jk_energy_per_atom', '2e derivatives (jk_energy_per_atom)', 'ejk1')
    patch(type(job), 'grad_nuc', 'nuclear repulsion (grad_nuc)')
elif args.what == 'tdgrad':
    td = mf.TDDFT() if args.rpa else mf.TDA()
    td.nstates = args.nstates
    td.kernel()
    job = td.nuc_grad_method()
    run = lambda: job.kernel(state=args.state)  # noqa: E731
    patch(tdrks_grad, '_contract_xc_kernel', 'XC kernel contractions (_contract_xc_kernel)')
    patch(cphf, 'solve', 'Z-vector CPHF (cphf.solve)')
    for name in ('jk_energies_per_atom', 'jk_energy_per_atom'):
        patch(type(job), name, f'2e derivatives ({name})', 'ejk1')
else:
    job = mf.Hessian()
    run = job.kernel
    patch(rhf_hess, '_partial_ejk_ip2', '2e second derivatives (_partial_ejk_ip2)')
    patch(rks_hess, '_get_exc_deriv2', 'XC second derivatives (_get_exc_deriv2)')
    patch(rks_hess, '_get_vxc_deriv1', 'XC first derivatives of V (_get_vxc_deriv1)')
    patch(rhf_hess, '_get_jk_ip1', 'J/K derivative matrices (_get_jk_ip1)')
    patch(rhf_hess, 'solve_mo1', 'CPHF (solve_mo1)')
    patch(type(job), 'get_jk_mo', 'CPHF J/K in the MO basis (get_jk_mo)')
    patch(rks_hess, 'nr_rks_fxc_mo', 'CPHF XC response (nr_rks_fxc_mo)')
patch(numint.NumInt, 'cache_xc_kernel', 'ground-state XC kernel (cache_xc_kernel)')
patch(j_engine._VHFOpt, 'get_j', 'J-engine builds (j_engine._VHFOpt.get_j)', 'jo')
patch(jk._VHFOpt, 'get_k', 'K builds (jk._VHFOpt.get_k)', 'ko')
for name in ('get_jk', 'get_j', 'get_k'):
    patch(type(mf), name, f'J/K builds ({name})', 'jk')
for name in ('nr_rks_fxc', 'nr_rks_fxc_st', 'nr_uks_fxc'):
    if hasattr(numint.NumInt, name):
        patch(numint.NumInt, name, f'XC response ({name})', 'fxc')
if args.what == 'grad' and args.first_stages:
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    run()
    cupy.cuda.Device().synchronize()
    first = time.perf_counter() - t0
    print(f'first call: {first:.2f} s')
    for label in sorted(times, key=lambda k: -times[k]):
        print(f'  {times[label]:8.2f} s {100 * times[label] / first:5.1f}%  {label}, {calls[label]} calls')
    times.clear()
    calls.clear()
with GpuMonitor() as mon:
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    run()
    cupy.cuda.Device().synchronize()
    total = time.perf_counter() - t0
title = f'{args.mol} {args.basis} {args.xc} ' + {
    'tdgrad': f'{"TDDFT" if args.rpa else "TDA"} gradient of state {args.state}', 'grad': 'gradient',
    'hess': 'Hessian'}[args.what]
print(f'{os.path.dirname(gpu4pyscf.__file__)}: {title}: {total:.2f} s' +
      (f' (first call {first:.2f} s)' if first else '') + f', SCF {mf.cycles} cycles, SM {mon.summary.get("sm_mhz")} MHz')
for label in sorted(times, key=lambda k: -times[k]):
    print(f'  {times[label]:8.2f} s {100 * times[label] / total:5.1f}%  {label}, {calls[label]} calls')
if args.out:
    with open(args.out, 'a') as f:
        f.write(json.dumps({'tree': os.path.dirname(gpu4pyscf.__file__), 'what': title, 'total_s': round(total, 3),
                            'first_s': round(first, 3) if first else None, 'scf_cycles': int(mf.cycles),
                            'stages': {k: [round(v, 3), calls[k]] for k, v in times.items()},
                            'gpu': mon.summary}) + '\n')
