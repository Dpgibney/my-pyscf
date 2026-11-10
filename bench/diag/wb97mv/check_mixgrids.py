import time, numpy
from pyscf import gto, dft
from pyscf.gto import moleintor
from molecules import CAFFEINE
print('libcint:', sorted({l.split()[-1] for l in open('/proc/self/maps') if 'libcint' in l}))
cf, cl, omega = 0.15, 0.85, 0.3
NPT = 301   # not a multiple of the SIMD width; arrays stay below ~0.3 GB each
for name, atom, basis, nsh in [('water/def2-qzvppd', 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', 'def2-qzvppd', None),
                               ('caffeine/def2-tzvpd, 100 shells', CAFFEINE, 'def2-tzvpd', 100),
                               ('HBr/cc-pvtz', 'H 0 0 0; Br 0 0 1.41', 'cc-pvtz', None)]:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    nsh = mol.nbas if nsh is None else nsh
    g = dft.gen_grid.Grids(mol); g.level = 0; g.build()
    coords = g.coords[::max(1, len(g.coords)//NPT)][:NPT]
    nao = mol.ao_loc[nsh]
    for intor, comp in (('int1e_grids_sph', 1), ('int1e_grids_ip_sph', 3)):
        assert comp * nao * nao * NPT * 8 < 1.2e9
        def ints(om, c13=0., c14=0., opt=None):
            with mol.with_range_coulomb(om):
                env = mol._env.copy()
            env[13], env[14] = c13, c14
            env = numpy.append(env, coords.ravel())
            env[gto.NGRIDS] = len(coords); env[gto.PTR_GRIDS] = mol._env.size
            kw = {} if opt is None else {'cintopt': opt}
            return moleintor.getints(intor, mol._atm, mol._bas, env, (0, nsh, 0, nsh, 0, len(coords)),
                                     comp=comp, aosym='s1', **kw)
        t = time.perf_counter(); full = ints(0.); t_full = time.perf_counter() - t
        t = time.perf_counter(); lr = ints(omega); t_lr = time.perf_counter() - t
        t = time.perf_counter(); mix = ints(omega, cf, cl); t_mix = time.perf_counter() - t
        scale = abs(full).max()
        err = abs(mix - (cf * full + cl * lr)).max() / scale
        e1 = abs(ints(omega, 1., 0.) - full).max() / scale
        e2 = abs(ints(omega, 0., 1.) - lr).max() / scale
        line = (f'{name} {intor} (nao {nao}): mixed vs {cf} full + {cl} LR {err:.1e}; (1,0) vs full {e1:.1e}; '
                f'(0,1) vs LR {e2:.1e}; t full {t_full:.2f} LR {t_lr:.2f} mixed {t_mix:.2f}')
        if comp == 1:
            with mol.with_range_coulomb(0.):
                opt = moleintor.make_cintopt(mol._atm, mol._bas, mol._env, intor)
            t = time.perf_counter(); f_o = ints(0., opt=opt); t_fo = time.perf_counter() - t
            t = time.perf_counter(); m_o = ints(omega, cf, cl, opt=opt); t_mo = time.perf_counter() - t
            line += f'; with optimizer: full {abs(f_o - full).max():.1e} ({t_fo:.2f} s), mixed {abs(m_o - mix).max():.1e} ({t_mo:.2f} s)'
        print(line, flush=True)
        full = lr = mix = None
