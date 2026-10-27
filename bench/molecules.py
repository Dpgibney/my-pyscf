"""Benchmark geometries (Angstrom)."""
import numpy as np

CAFFEINE = '''
C  1.5737 -0.4553 0.0000; N  0.3071 -0.9879 0.0000; C -0.7612 -0.1093 0.0000; N -0.4795  1.2263 0.0000
C  0.8099  1.7409 0.0000; C  1.8839  0.8780 0.0000; O  2.4789 -1.2775 0.0000; O -1.8901 -0.5617 0.0000
N  1.0247  3.0786 0.0000; C  2.3466  3.2127 0.0000; N  2.9963  2.0215 0.0000; C  0.1124 -2.4295 0.0000
C -1.5776  2.1878 0.0000; C  4.4256  1.8044 0.0000; H  2.8297  4.1757 0.0000; H  0.7330 -2.9019 0.8857
H  0.7330 -2.9019 -0.8857; H -0.9372 -2.6996 0.0000; H -2.5313 1.6662 0.0000; H -1.5050  2.8303 0.8823
H -1.5050  2.8303 -0.8823; H  4.6830 1.2259 0.8917; H  4.6830 1.2259 -0.8917; H  4.9022  2.7808 0.0000'''

BENZENE = '''
C  0.0000  1.3970 0; C  1.2098  0.6985 0; C  1.2098 -0.6985 0
C  0.0000 -1.3970 0; C -1.2098 -0.6985 0; C -1.2098  0.6985 0
H  0.0000  2.4870 0; H  2.1538  1.2435 0; H  2.1538 -1.2435 0
H  0.0000 -2.4870 0; H -2.1538 -1.2435 0; H -2.1538  1.2435 0'''

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def n_alkane(n, r_cc=1.53, angle_ccc=112.0, r_ch=1.09):
    '''All-trans CnH(2n+2) zigzag in the xy-plane with tetrahedral hydrogens.'''
    half = np.radians(angle_ccc) / 2
    a, b = r_cc * np.sin(half), r_cc * np.cos(half)
    # positions -1 and n are the virtual neighbours that place the terminal hydrogens
    backbone = {i: np.array([i * a, (i % 2) * b, 0.]) for i in range(-1, n + 1)}
    phi = np.radians(109.47) / 2
    z = np.array([0., 0., 1.])
    atoms = [('C', backbone[i]) for i in range(n)]
    for i in range(n):
        c = backbone[i]
        u1 = backbone[i - 1] - c
        u2 = backbone[i + 1] - c
        w = -(u1 / np.linalg.norm(u1) + u2 / np.linalg.norm(u2))
        w /= np.linalg.norm(w)
        for s in (1, -1):
            atoms.append(('H', c + r_ch * (np.cos(phi) * w + s * np.sin(phi) * z)))
        if i in (0, n - 1):  # methyl: third H in-plane, where the missing carbon would be
            v = (u1 if i == 0 else u2)
            atoms.append(('H', c + r_ch * v / np.linalg.norm(v)))
    return '; '.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in atoms)
