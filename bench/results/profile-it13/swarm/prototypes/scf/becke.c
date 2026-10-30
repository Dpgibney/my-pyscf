/* VXCgen_grid of pyscf/lib/dft/grid_basis.c, verbatim (renamed), compiled with different -m flags,
 * plus BECKEweights: the same pair loop, but it returns the weights vol*P[ia]/sum_B P[B] directly
 * (the normalization sum taken over atoms in order 0..natm-1, as numpy's pbecke.sum(axis=0)). */
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define GRIDS_BLOCK     512
#define ALIGNMENT       8

void BECKEgen_grid(double *out, double *coords, double *atm_coords,
                   double *radii_table, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i, j;
        double dx, dy, dz;
        double *atom_dist = malloc(sizeof(double) * natm*natm);
        for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        dx = atm_coords[i*3+0] - atm_coords[j*3+0];
                        dy = atm_coords[i*3+1] - atm_coords[j*3+1];
                        dz = atm_coords[i*3+2] - atm_coords[j*3+2];
                        atom_dist[i*natm+j] = 1 / sqrt(dx*dx + dy*dy + dz*dz);
                }
        }

#pragma omp parallel private(i, j, dx, dy, dz)
{
        double *_buf = malloc(sizeof(double) * ((natm*2+1)*GRIDS_BLOCK + ALIGNMENT));
        double *buf = (double *)((uintptr_t)(_buf + ALIGNMENT - 1) & (-(uintptr_t)(ALIGNMENT*8)));
        double *g = buf + natm * GRIDS_BLOCK;
        double *grid_dist = g + GRIDS_BLOCK;
        size_t ig0, n, ngs;
        double fac, s;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
                ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
                for (i = 0; i < natm; i++) {
#pragma GCC ivdep
                for (n = 0; n < ngs; n++) {
                        dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                        dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                        dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                        grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                        buf[i*GRIDS_BLOCK+n] = 1;
                } }

                for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {

                        fac = atom_dist[i*natm+j];
                        for (n = 0; n < ngs; n++) {
                                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                                        grid_dist[j*GRIDS_BLOCK+n]) * fac;
                        }
                        if (radii_table != NULL) {
                                fac = radii_table[i*natm+j];
                                for (n = 0; n < ngs; n++) {
                                        g[n] += fac * (1 - g[n]*g[n]);
                                }
                        }
#pragma GCC ivdep
                        for (n = 0; n < ngs; n++) {
                                s = g[n];
                                s = (3 - s*s) * s * .5;
                                s = (3 - s*s) * s * .5;
                                s = ((3 - s*s) * s * .5) * .5;
                                buf[i*GRIDS_BLOCK+n] *= .5 - s;
                                buf[j*GRIDS_BLOCK+n] *= .5 + s;
                        }
                } }

                for (i = 0; i < natm; i++) {
                        for (n = 0; n < ngs; n++) {
                                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                        }
                }
        }
        free(_buf);
}
        free(atom_dist);
}

/* weights[n] = vol[n] * P_ia[n] * (1. / sum_B P_B[n]), the pair loop fused (g, radii adjustment and
 * the switching function in one pass, same operations per element in the same order) */
void BECKEweights(double *weights, double *vol, int ia, double *coords, double *atm_coords,
                  double *radii_table, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i, j;
        double dx, dy, dz;
        double *atom_dist = malloc(sizeof(double) * natm*natm);
        for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        dx = atm_coords[i*3+0] - atm_coords[j*3+0];
                        dy = atm_coords[i*3+1] - atm_coords[j*3+1];
                        dz = atm_coords[i*3+2] - atm_coords[j*3+2];
                        atom_dist[i*natm+j] = 1 / sqrt(dx*dx + dy*dy + dz*dz);
                }
        }
#pragma omp parallel private(i, j, dx, dy, dz)
{
        double *_buf = malloc(sizeof(double) * ((natm+1)*GRIDS_BLOCK + ALIGNMENT));
        double *buf = (double *)((uintptr_t)(_buf + ALIGNMENT - 1) & (-(uintptr_t)(ALIGNMENT*8)));
        double *grid_dist = buf + natm * GRIDS_BLOCK;
        double *_gd = malloc(sizeof(double) * (natm*GRIDS_BLOCK + ALIGNMENT));
        double *gd = (double *)((uintptr_t)(_gd + ALIGNMENT - 1) & (-(uintptr_t)(ALIGNMENT*8)));
        size_t ig0, n, ngs;
        double fac, fac2, s, sum;
        (void)grid_dist;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
                ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
                for (i = 0; i < natm; i++) {
#pragma GCC ivdep
                for (n = 0; n < ngs; n++) {
                        dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                        dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                        dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                        gd[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                        buf[i*GRIDS_BLOCK+n] = 1;
                } }
                for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        fac = atom_dist[i*natm+j];
                        double *bi = buf + i*GRIDS_BLOCK;
                        double *bj = buf + j*GRIDS_BLOCK;
                        double *di = gd + i*GRIDS_BLOCK;
                        double *dj = gd + j*GRIDS_BLOCK;
                        if (radii_table != NULL) {
                                fac2 = radii_table[i*natm+j];
#pragma GCC ivdep
                                for (n = 0; n < ngs; n++) {
                                        s = (di[n] - dj[n]) * fac;
                                        s += fac2 * (1 - s*s);
                                        s = (3 - s*s) * s * .5;
                                        s = (3 - s*s) * s * .5;
                                        s = ((3 - s*s) * s * .5) * .5;
                                        bi[n] *= .5 - s;
                                        bj[n] *= .5 + s;
                                }
                        } else {
#pragma GCC ivdep
                                for (n = 0; n < ngs; n++) {
                                        s = (di[n] - dj[n]) * fac;
                                        s = (3 - s*s) * s * .5;
                                        s = (3 - s*s) * s * .5;
                                        s = ((3 - s*s) * s * .5) * .5;
                                        bi[n] *= .5 - s;
                                        bj[n] *= .5 + s;
                                }
                        }
                } }
                for (n = 0; n < ngs; n++) {
                        sum = buf[n];
                        for (i = 1; i < natm; i++) {
                                sum += buf[i*GRIDS_BLOCK+n];
                        }
                        weights[ig0+n] = vol[ig0+n] * buf[ia*GRIDS_BLOCK+n] * (1. / sum);
                }
        }
        free(_buf);
        free(_gd);
}
        free(atom_dist);
}
