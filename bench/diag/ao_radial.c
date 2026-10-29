/* AO-cache layouts for ao_radial_cache.py.
 * A spherical AO is a polynomial in (r - R_atom) times its shell's contracted radial factor
 * e_k(r) = sum_p c_kp exp(-a_p r^2); first derivatives also need e2a_k(r) = sum_p -2 a_p c_kp exp(-a_p r^2).
 * GTOcontract_exp0/1 compute these per 56-point block, GTOshell_eval_grid_cart(_deriv1) and
 * CINTc2s_ket_sph1 turn them into AO values. For every block and unscreened shell:
 *   RAD_FILL  stores e (and e2a for deriv 1): (deriv+1)*nctr doubles per point
 *   RAD_EVAL  rebuilds the AO values from them with PySCF's own polynomial and c2s code
 *   FULL_FILL stores the finished AO values: ncomp*nctr*(2l+1) doubles per point
 *   FULL_EVAL copies them back
 *   STOCK_EVAL computes them as GTOeval_sph_iter does (exp, contraction, polynomial, c2s)
 * The EVAL modes write the same dense block as GTOval_sph_deriv0/1 (zeros for screened
 * shells), so RAD_EVAL equals eval_ao bit for bit. AOcache_drv makes one block_loop block with
 * eval_ao's OpenMP work split (56-point block x atom); AOcache_fused makes every 56-point block
 * of the grid in a per-thread buffer that stays in cache, as a numint that consumed each block right
 * away would, and throws it away.
 * Build (flags of libcgto):
 *   L=pyscf/pyscf/lib; gcc -msse3 -O2 -std=gnu99 -fPIC -fopenmp -shared -I$L -I$L/gto \
 *       -I$L/deps/include -o bench/diag/ao_radial.so bench/diag/ao_radial.c \
 *       -L$L -L$L/deps/lib -lcgto -lcint -Wl,-rpath,$PWD/$L:$PWD/$L/deps/lib
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "config.h"
#include "grid_ao_drv.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))

enum { RAD_FILL, RAD_EVAL, FULL_FILL, FULL_EVAL, STOCK_EVAL };

double CINTcommon_fac_sp(int l);
double *CINTc2s_ket_sph1(double *sph, double *cart, int lds, int ldc, int l);
int GTOshloc_by_atom(int *shloc, int *shls_slice, int *ao_loc, int *atm, int *bas);
int GTOcontract_exp0(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac);
int GTOcontract_exp1(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac);
void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
                             double *coord, double *alpha, double *coeff,
                             double *env, int l, int np, int nc,
                             size_t nao, size_t ngrids, size_t blksize);
void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t blksize);

/* One 56-point block of the shells of one atom, as GTOeval_sph_iter. ao points at the first
 * function of shell sh0 and grid point 0 of the block, ngrids is its row stride and ldc the
 * stride of coord; cache_off[sh] (cache_off = offs of this block) is where shell sh's data
 * starts in cache. */
static void _iter(int mode, int deriv, size_t nao, size_t ngrids, size_t ldc, size_t bgrids,
                  int sh0, int sh1, int *ao_loc, double *buf, double *ao, double *coord,
                  uint8_t *non0table, double *cache, int64_t *cache_off,
                  int *atm, int *bas, double *env)
{
        const int ncomp = deriv == 0 ? 1 : 4;
        FPtr_exp fexp = deriv == 0 ? GTOcontract_exp0 : GTOcontract_exp1;
        FPtr_eval feval = deriv == 0 ? GTOshell_eval_grid_cart : GTOshell_eval_grid_cart_deriv1;
        const int atm_id = bas[sh0*BAS_SLOTS+ATOM_OF];
        double *ri = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
        double *grid2atm = ALIGN8_UP(buf);  // [xyz,grid] of this atom
        double *eprim = grid2atm + 3*BLKSIZE;
        double *cart_gto = eprim + NPRIMAX*BLKSIZE*2;
        int i, k, r, l, np, nc, deg, dcart, bas_id;
        size_t ig, ao_id, di;
        double *p_exp, *pcoeff, *pcart, *pao, *pc;

        if (mode == RAD_FILL || mode == RAD_EVAL || mode == STOCK_EVAL) {
                for (ig = 0; ig < bgrids; ig++) {
                        grid2atm[0*BLKSIZE+ig] = coord[0*ldc+ig] - ri[0];
                        grid2atm[1*BLKSIZE+ig] = coord[1*ldc+ig] - ri[1];
                        grid2atm[2*BLKSIZE+ig] = coord[2*ldc+ig] - ri[2];
                }
        }

        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                np = bas[bas_id*BAS_SLOTS+NPRIM_OF];
                nc = bas[bas_id*BAS_SLOTS+NCTR_OF ];
                l  = bas[bas_id*BAS_SLOTS+ANG_OF  ];
                deg = l * 2 + 1;
                dcart = (l+1)*(l+2)/2;
                p_exp  = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
                pcoeff = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
                ao_id = ao_loc[bas_id] - ao_loc[sh0];
                pc = cache + cache_off[bas_id];
                if (!non0table[bas_id]) {
                        if (mode != RAD_FILL && mode != FULL_FILL) {
                                for (i = 0; i < ncomp; i++) {
                                for (r = 0; r < nc*deg; r++) {
                                        memset(ao+(i*nao+ao_id+r)*ngrids, 0, sizeof(double)*bgrids);
                                } }
                        }
                        continue;
                }

                switch (mode) {
                case RAD_FILL:
                        (*fexp)(eprim, grid2atm, p_exp, pcoeff, l, np, nc, bgrids,
                                CINTcommon_fac_sp(l));
                        memcpy(pc, eprim, sizeof(double)*nc*BLKSIZE);
                        if (deriv == 1) {
                                memcpy(pc+nc*BLKSIZE, eprim+NPRIMAX*BLKSIZE, sizeof(double)*nc*BLKSIZE);
                        }
                        break;
                case RAD_EVAL:
                case STOCK_EVAL:
                        if (mode == STOCK_EVAL) {
                                (*fexp)(eprim, grid2atm, p_exp, pcoeff, l, np, nc, bgrids,
                                        CINTcommon_fac_sp(l));
                        } else {
                                memcpy(eprim, pc, sizeof(double)*nc*BLKSIZE);
                                if (deriv == 1) {
                                        memcpy(eprim+NPRIMAX*BLKSIZE, pc+nc*BLKSIZE,
                                               sizeof(double)*nc*BLKSIZE);
                                }
                        }
                        if (l <= 1) {
                                (*feval)(ao+ao_id*ngrids, ri, eprim, grid2atm, p_exp, pcoeff,
                                         env, l, np, nc, nao, ngrids, bgrids);
                        } else {
                                di = nc * dcart;
                                (*feval)(cart_gto, ri, eprim, grid2atm, p_exp, pcoeff,
                                         env, l, np, nc, di, bgrids, bgrids);
                                pcart = cart_gto;
                                for (i = 0; i < ncomp; i++) {
                                        pao = ao + (i*nao+ao_id)*ngrids;
                                        for (k = 0; k < nc; k++) {
                                                CINTc2s_ket_sph1(pao, pcart, ngrids, bgrids, l);
                                                pao += deg * ngrids;
                                                pcart += dcart * bgrids;
                                        }
                                }
                        }
                        break;
                case FULL_FILL:
                case FULL_EVAL:
                        for (i = 0; i < ncomp; i++) {
                        for (r = 0; r < nc*deg; r++) {
                                pao = ao + (i*nao+ao_id+r)*ngrids;
                                if (mode == FULL_FILL) {
                                        memcpy(pc, pao, sizeof(double)*bgrids);
                                } else {
                                        memcpy(pao, pc, sizeof(double)*bgrids);
                                }
                                pc += BLKSIZE;
                        } }
                }
        }
}

/* One block_loop block of ngrids points: ao[ncomp,nao,ngrids], coord[3,ngrids],
 * non0table[nblk,nbas] and offs[nblk,nbas] start at this block's first 56-point block. */
void AOcache_drv(int mode, int deriv, int ngrids, double *ao, double *cache, int64_t *offs,
                 double *coord, uint8_t *non0table, int *ao_loc,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        int shls_slice[2] = {0, nbas};
        int shloc[nbas+1];
        const int nshblk = GTOshloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        const size_t nao = ao_loc[nbas];
        const size_t Ngrids = ngrids;
#pragma omp parallel
{
        int ib, k, iloc;
        size_t ip, bgrids;
        double *buf = malloc(sizeof(double) * BLKSIZE*(NPRIMAX*2+NCTR_CART*4+4) + 64);
#pragma omp for schedule(dynamic, 4)
        for (k = 0; k < nblk*nshblk; k++) {
                iloc = k / nblk;
                ib = k - iloc * nblk;
                ip = (size_t)ib * BLKSIZE;
                bgrids = MIN(Ngrids-ip, BLKSIZE);
                _iter(mode, deriv, nao, Ngrids, Ngrids, bgrids, shloc[iloc], shloc[iloc+1], ao_loc, buf,
                      ao+(size_t)ao_loc[shloc[iloc]]*Ngrids+ip, coord+ip,
                      non0table+(size_t)ib*nbas, cache, offs+(size_t)ib*nbas,
                      atm, bas, env);
        }
        free(buf);
}
}

/* Every 56-point block of the whole grid (coord[3,ngrids], non0table and offs [nblk,nbas]),
 * each into the calling thread's buffer ao[ncomp,nao,56]. */
void AOcache_fused(int mode, int deriv, int ngrids, double *cache, int64_t *offs,
                   double *coord, uint8_t *non0table, int *ao_loc,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        int shls_slice[2] = {0, nbas};
        int shloc[nbas+1];
        const int nshblk = GTOshloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        const size_t nao = ao_loc[nbas];
        const size_t Ngrids = ngrids;
        const int ncomp = deriv == 0 ? 1 : 4;
#pragma omp parallel
{
        int ib, iloc;
        size_t ip, bgrids;
        double *buf = malloc(sizeof(double) * BLKSIZE*(NPRIMAX*2+NCTR_CART*4+4) + 64);
        double *ao = malloc(sizeof(double) * ncomp*nao*BLKSIZE);
#pragma omp for schedule(dynamic, 4)
        for (ib = 0; ib < nblk; ib++) {
                ip = (size_t)ib * BLKSIZE;
                bgrids = MIN(Ngrids-ip, BLKSIZE);
                for (iloc = 0; iloc < nshblk; iloc++) {
                        _iter(mode, deriv, nao, BLKSIZE, Ngrids, bgrids, shloc[iloc], shloc[iloc+1],
                              ao_loc, buf, ao+(size_t)ao_loc[shloc[iloc]]*BLKSIZE, coord+ip,
                              non0table+(size_t)ib*nbas, cache, offs+(size_t)ib*nbas,
                              atm, bas, env);
                }
        }
        free(ao);
        free(buf);
}
}
