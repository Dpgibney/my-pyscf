# rot_smd.py's SMD rows with coarser and finer surfaces: the spread of the frequencies over rigid rotations against
# with_solvent.lebedev_order (SMD's default is 41, 590 points per atomic sphere before the switching; PCM's is 29, 302)
# Run both from this directory (the geometries are in cpu_ts/): python rot_smd.py; python rot_smd_leb.py
import numpy
from scipy.spatial.transform import Rotation
import rot_smd_lib as L
sym, x0 = L.geom('cpu_ts/hcn_smd')
rots = [numpy.eye(3)] + [Rotation.random(random_state=seed).as_matrix() for seed in (1, 2, 3)]
for order in (29, 41, 53, 65):
    rows = []
    for R in rots:
        e, g, f, n = L.freqs(sym, x0.dot(R.T), True, order)
        rows.append([e] + list(f))
    rows = numpy.array(rows)
    print('lebedev_order %d (%d surface points): E spread %.1e Eh; frequencies (mean, max - min): %s' % (
        order, n, numpy.ptp(rows[:,0]),
        '  '.join('%.1f %.2f' % (rows[:,k].mean(), numpy.ptp(rows[:,k])) for k in range(1, 4))), flush=True)
