'''Runs bench/diag/rijcosx.py (arguments passed through) with nr_nlc_vxc on its
block_loop path: the fused XC kernels off during the VV10 call only, as the
tree was before VV10 took the fused kernels.'''
import os, sys, runpy
from pyscf.dft import numint

harness = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rijcosx.py')
orig = numint.NumInt.nr_nlc_vxc

def nr_nlc_vxc(self, *a, **k):
    numint.FUSED_XC = False
    try:
        return orig(self, *a, **k)
    finally:
        numint.FUSED_XC = True

numint.NumInt.nr_nlc_vxc = nr_nlc_vxc
sys.argv = [harness] + sys.argv[1:]
runpy.run_path(harness, run_name='__main__')
