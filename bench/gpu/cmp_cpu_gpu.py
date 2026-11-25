'''GPU (prod_parts.py --save) against CPU (cpu_ref.py) results of one molecule: energy, gradient, Hessian, frequencies.
Usage: cmp_cpu_gpu.py XYZ SPIN BASIS CPU.npz GPU.npz'''
import sys, numpy
from pyscf import gto
from pyscf.hessian import thermo
mol = gto.M(atom=sys.argv[1], basis=sys.argv[3], spin=int(sys.argv[2]), verbose=0)
a, b = numpy.load(sys.argv[4]), numpy.load(sys.argv[5])
fa = thermo.harmonic_analysis(mol, a['hess'], imaginary_freq=False)['freq_wavenumber']
fb = thermo.harmonic_analysis(mol, b['hess'], imaginary_freq=False)['freq_wavenumber']
print(f'{sys.argv[5]}: E(GPU) - E(CPU) = {float(b["e"]) - float(a["e"]):+.2e} Eh; max |d grad| {abs(a["grad"] - b["grad"]).max():.1e} '
      f'(max |grad| {abs(a["grad"]).max():.2e}); max |d hess| {abs(a["hess"] - b["hess"]).max():.1e}; '
      f'max |d freq| {abs(fa - fb).max():.3f} cm^-1 (lowest {fa[0]:.1f} / {fb[0]:.1f})')
