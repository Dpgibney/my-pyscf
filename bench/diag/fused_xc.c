/* Prototype for fused_xc.py: numint's sparse XC kernels with the AO values computed in
 * place. VXCeval_rho_sparse_gemm, VXCdot_ao_aow_sparse_gemm and VXCdot_ao_wv_ao_sparse_gemm
 * (pyscf lib/dft/nr_numint_sparse.c, It6) gather, for each group of 4 blocks of 56 grid
 * points, the AO values of the shells unscreened on any of the blocks from the AO block that
 * eval_ao wrote to memory. The FUSED* kernels below compute those values directly in the
 * gather buffers instead, with the GTO code eval_ao uses (GTOcontract_exp0/1,
 * GTOshell_eval_grid_cart(_deriv1), CINTc2s_ket_sph1), so no AO block is written or read.
 * With rcache, the shells' contracted radial factors (as ao_radial.c's radial cache) are
 * read instead of computed. The same values then go through the same dgemm/dsyrk calls and
 * sums, so the results equal the It6 kernels' bit for bit. The group machinery is copied
 * from nr_numint_sparse.c.
 * Build:
 *   L=pyscf/pyscf/lib; gcc -msse3 -O2 -std=gnu99 -fPIC -fopenmp -shared -I$L -I$L/deps/include \
 *       -o bench/diag/fused_xc.so bench/diag/fused_xc.c -L$L -L$L/deps/lib \
 *       -lcgto -lcint -lnp_helper -lopenblas -Wl,-rpath,$PWD/$L:$PWD/$L/deps/lib
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

#define BLKSIZE         56      // gto/grid_ao_drv.h
#define NPRIMAX         40      // gto/grid_ao_drv.h
#define NCTR_CART       128     // gto/grid_ao_drv.h
#define GEMM_NBLK       4
#define GEMM_NGRIDS     (GEMM_NBLK*BLKSIZE)
#define EVAL_WORK       (BLKSIZE*(3 + NPRIMAX*2 + NCTR_CART*4))

typedef int (*FPtr_exp)(double *ectr, double *coord, double *alpha, double *coeff,
                        int l, int nprim, int nctr, size_t ngrids, double fac);
typedef void (*FPtr_eval)(double *gto, double *ri, double *exps,
                          double *coord, double *alpha, double *coeff,
                          double *env, int l, int np, int nc,
                          size_t nao, size_t ngrids, size_t blksize);
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
double *CINTc2s_ket_sph1(double *sph, double *cart, int lds, int ldc, int l);
double CINTcommon_fac_sp(int l);

/* ---- copied from nr_numint_sparse.c ---- */
typedef struct {
        int nao;            // number of AOs of the unscreened shells
        int nshl;           // number of unscreened shells
        int nseg;           // number of contiguous AO ranges they form
        int *shl;           // [nshl] shell ids, ascending
        uint8_t *blkmask;   // [nshl] bit b: the shell is unscreened on block blk0+b
        int *seg_ao;        // [nseg] first AO of each range
        int *seg_off;       // [nseg] position of that AO in the gathered buffers
        int *seg_len;       // [nseg] number of AOs in the range
} GemmGroup;

static void _gemm_group_init(GemmGroup *g, int nbas)
{
        g->shl = malloc(sizeof(int) * nbas * 4);
        g->seg_ao = g->shl + nbas;
        g->seg_off = g->seg_ao + nbas;
        g->seg_len = g->seg_off + nbas;
        g->blkmask = malloc(sizeof(uint8_t) * nbas);
}

static void _gemm_group_free(GemmGroup *g)
{
        free(g->shl);
        free(g->blkmask);
}

static void _gemm_group_shells(GemmGroup *g, uint8_t *screen_index,
                               int blk0, int blk1, int nbas, int *ao_loc)
{
        int ish, b, mask;
        int nao = 0;
        int nshl = 0;
        int nseg = 0;
        for (ish = 0; ish < nbas; ish++) {
                mask = 0;
                for (b = 0; b < blk1 - blk0; b++) {
                        if (screen_index[(blk0+b)*nbas+ish]) {
                                mask |= 1 << b;
                        }
                }
                if (mask == 0) {
                        continue;
                }
                if (nseg == 0 ||
                    g->seg_ao[nseg-1] + g->seg_len[nseg-1] != ao_loc[ish]) {
                        g->seg_ao[nseg] = ao_loc[ish];
                        g->seg_off[nseg] = nao;
                        g->seg_len[nseg] = 0;
                        nseg++;
                }
                g->seg_len[nseg-1] += ao_loc[ish+1] - ao_loc[ish];
                g->shl[nshl] = ish;
                g->blkmask[nshl] = mask;
                nshl++;
                nao += ao_loc[ish+1] - ao_loc[ish];
        }
        g->nao = nao;
        g->nshl = nshl;
        g->nseg = nseg;
}

static int _gemm_max_group_aos(uint8_t *screen_index, int ngrids, int nbas,
                               int *ao_loc)
{
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        int max_nao = 0;
        int igroup;
#pragma omp parallel for schedule(static) reduction(max:max_nao)
        for (igroup = 0; igroup < ngroup; igroup++) {
                int blk0 = igroup * GEMM_NBLK;
                int blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                int nao = 0;
                int ish, b;
                for (ish = 0; ish < nbas; ish++) {
                        for (b = blk0; b < blk1; b++) {
                                if (screen_index[b*nbas+ish]) {
                                        nao += ao_loc[ish+1] - ao_loc[ish];
                                        break;
                                }
                        }
                }
                max_nao = MAX(max_nao, nao);
        }
        return max_nao;
}

static void _gemm_scatter_add(double *out, double *cbuf, GemmGroup *g,
                              size_t Nao, int tril)
{
        const int ns = g->nao;
        int s, s1, i, m, m0, mlen;
        int n = 0;
        double *pout, *pc;
        for (s = 0; s < g->nseg; s++) {
        for (i = g->seg_ao[s]; i < g->seg_ao[s] + g->seg_len[s]; i++, n++) {
                pc = cbuf + n * (size_t)ns;
                pout = out + i * Nao;
                for (s1 = 0; s1 < g->nseg; s1++) {
                        m0 = g->seg_off[s1];
                        mlen = g->seg_len[s1];
                        if (tril) {
                                if (m0 > n) {
                                        break;
                                }
                                mlen = MIN(mlen, n - m0 + 1);
                        }
                        for (m = 0; m < mlen; m++) {
                                pout[g->seg_ao[s1]+m] += pc[m0+m];
                        }
                }
        } }
}

static void _gemm_contract_buf(double *rho, double *a, double *c, int ns, int ng)
{
        int n, k;
        double *pa, *pc;
        for (k = 0; k < ng; k++) {
                rho[k] = 0;
        }
        for (n = 0; n < ns; n++) {
                pa = a + n * (size_t)ng;
                pc = c + n * (size_t)ng;
                for (k = 0; k < ng; k++) {
                        rho[k] += pa[k] * pc[k];
                }
        }
}
/* ---- end of copy ---- */

/* aobuf[c,n,k] for c < ncomp (1 for deriv 0, 4 for deriv 1), n < g->nao, k < ng: the AO
 * value (and gradient) of the n-th AO of group g at grid point blk0*BLKSIZE+k of the
 * block_loop block whose coordinates are coords[3,ldc], as eval_ao computes it; zero on
 * the blocks where its shell is screened. With rcache, the radial factors of block b and
 * shell i are read from rcache + roffs[b*nbas+i]. */
static void _eval_group(double *aobuf, int deriv, GemmGroup *g, int ng,
                        double *coords, size_t ldc, int blk0,
                        double *rcache, int64_t *roffs, int nbas,
                        int *atm, int *bas, double *env, double *work)
{
        const int ncomp = deriv == 0 ? 1 : 4;
        const size_t ns = g->nao;
        const size_t comp_size = ns * ng;
        const size_t ig0 = (size_t)blk0 * BLKSIZE;
        FPtr_exp fexp = deriv == 0 ? GTOcontract_exp0 : GTOcontract_exp1;
        FPtr_eval feval = deriv == 0 ? GTOshell_eval_grid_cart : GTOshell_eval_grid_cart_deriv1;
        double *grid2atm = work;
        double *eprim = grid2atm + 3*BLKSIZE;
        double *cart = eprim + NPRIMAX*BLKSIZE*2;
        int s, b, k0, k, r, c, ic, ish, l, np, nc, deg, dcart, bgrids;
        size_t n = 0;
        double *ri, *p_exp, *pcoeff, *pao, *pcart, *pc, *px;
        for (s = 0; s < g->nshl; s++) {
                ish = g->shl[s];
                l  = bas[ish*BAS_SLOTS+ANG_OF];
                np = bas[ish*BAS_SLOTS+NPRIM_OF];
                nc = bas[ish*BAS_SLOTS+NCTR_OF];
                deg = 2 * l + 1;
                dcart = (l+1)*(l+2)/2;
                ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
                p_exp  = env + bas[ish*BAS_SLOTS+PTR_EXP];
                pcoeff = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                        bgrids = MIN(BLKSIZE, ng - k0);
                        pao = aobuf + n * ng + k0;
                        if (!(g->blkmask[s] & (1 << b))) {
                                for (c = 0; c < ncomp; c++) {
                                for (r = 0; r < nc*deg; r++) {
                                        px = pao + c * comp_size + r * (size_t)ng;
                                        for (k = 0; k < bgrids; k++) {
                                                px[k] = 0;
                                        }
                                } }
                                continue;
                        }
                        px = coords + ig0 + k0;
                        for (k = 0; k < bgrids; k++) {
                                grid2atm[0*BLKSIZE+k] = px[0*ldc+k] - ri[0];
                                grid2atm[1*BLKSIZE+k] = px[1*ldc+k] - ri[1];
                                grid2atm[2*BLKSIZE+k] = px[2*ldc+k] - ri[2];
                        }
                        if (rcache != NULL) {
                                pc = rcache + roffs[(size_t)(blk0+b)*nbas+ish];
                                memcpy(eprim, pc, sizeof(double)*nc*BLKSIZE);
                                if (deriv == 1) {
                                        memcpy(eprim+NPRIMAX*BLKSIZE, pc+nc*BLKSIZE,
                                               sizeof(double)*nc*BLKSIZE);
                                }
                        } else {
                                (*fexp)(eprim, grid2atm, p_exp, pcoeff, l, np, nc, bgrids,
                                        CINTcommon_fac_sp(l));
                        }
                        if (l <= 1) {
                                (*feval)(pao, ri, eprim, grid2atm, p_exp, pcoeff,
                                         env, l, np, nc, ns, ng, bgrids);
                        } else {
                                (*feval)(cart, ri, eprim, grid2atm, p_exp, pcoeff,
                                         env, l, np, nc, nc*dcart, bgrids, bgrids);
                                pcart = cart;
                                for (c = 0; c < ncomp; c++) {
                                for (ic = 0; ic < nc; ic++) {
                                        CINTc2s_ket_sph1(pao + c * comp_size + ic * deg * (size_t)ng,
                                                         pcart, ng, bgrids, l);
                                        pcart += dcart * bgrids;
                                } }
                        }
                }
                n += nc * deg;
        }
}

/* The radial factors of every 56-point block b and shell i unscreened on it (deriv 0:
 * nctr doubles per point, deriv 1: 2*nctr) at rcache + roffs[b*nbas+i]; coords[3,ngrids]. */
void FUSEDradial_fill(double *rcache, int64_t *roffs, int deriv, int ngrids, int nbas,
                      uint8_t *screen_index, double *coords,
                      int *atm, int *bas, double *env)
{
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        FPtr_exp fexp = deriv == 0 ? GTOcontract_exp0 : GTOcontract_exp1;
#pragma omp parallel
{
        double *grid2atm = malloc(sizeof(double) * BLKSIZE*(3 + NPRIMAX*2));
        double *eprim = grid2atm + 3*BLKSIZE;
        int ib, ish, k, l, nc, bgrids;
        double *ri, *pc, *px;
#pragma omp for schedule(dynamic, 4)
        for (ib = 0; ib < nblk; ib++) {
                bgrids = MIN(BLKSIZE, ngrids - ib*BLKSIZE);
                for (ish = 0; ish < nbas; ish++) {
                        if (!screen_index[(size_t)ib*nbas+ish]) {
                                continue;
                        }
                        l  = bas[ish*BAS_SLOTS+ANG_OF];
                        nc = bas[ish*BAS_SLOTS+NCTR_OF];
                        ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
                        px = coords + (size_t)ib * BLKSIZE;
                        for (k = 0; k < bgrids; k++) {
                                grid2atm[0*BLKSIZE+k] = px[0*(size_t)ngrids+k] - ri[0];
                                grid2atm[1*BLKSIZE+k] = px[1*(size_t)ngrids+k] - ri[1];
                                grid2atm[2*BLKSIZE+k] = px[2*(size_t)ngrids+k] - ri[2];
                        }
                        (*fexp)(eprim, grid2atm, env + bas[ish*BAS_SLOTS+PTR_EXP],
                                env + bas[ish*BAS_SLOTS+PTR_COEFF], l,
                                bas[ish*BAS_SLOTS+NPRIM_OF], nc, bgrids, CINTcommon_fac_sp(l));
                        pc = rcache + roffs[(size_t)ib*nbas+ish];
                        memcpy(pc, eprim, sizeof(double)*nc*BLKSIZE);
                        if (deriv == 1) {
                                memcpy(pc+nc*BLKSIZE, eprim+NPRIMAX*BLKSIZE, sizeof(double)*nc*BLKSIZE);
                        }
                }
        }
        free(grid2atm);
}
}

// rho[k] = sum_n a[n,k] * c[n,k] over the blocks where the shell of n is unscreened
// (the sums of _gemm_contract_rho, from the compact buffer a)
static void _contract_group(double *rho, double *a, double *c, GemmGroup *g,
                            int *ao_loc, int ng)
{
        int s, i, b, k, k0, k1;
        int n = 0;
        double *pa, *pc;
        for (k = 0; k < ng; k++) {
                rho[k] = 0;
        }
        for (s = 0; s < g->nshl; s++) {
        for (i = ao_loc[g->shl[s]]; i < ao_loc[g->shl[s]+1]; i++, n++) {
                pa = a + n * (size_t)ng;
                pc = c + n * (size_t)ng;
                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                        if (g->blkmask[s] & (1 << b)) {
                                k1 = MIN(k0 + BLKSIZE, ng);
                                for (k = k0; k < k1; k++) {
                                        rho[k] += pa[k] * pc[k];
                                }
                        }
                }
        } }
}

/* VXCeval_rho_sparse_gemm with the AO values computed in place: rho[nrho,ngrids] of a
 * real symmetric dm[nao,nao] for the block_loop block with coordinates coords[3,ldc]. */
void FUSEDeval_rho(double *rho, double *dm, int nderiv, int nao, int ngrids, int nbas,
                   uint8_t *screen_index, int *ao_loc, double *coords, int ldc,
                   int *atm, int *bas, double *env, double *rcache, int64_t *roffs)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const int deriv = nderiv == 0 ? 0 : 1;
        const int ncomp = deriv == 0 ? 1 : 4;
        const int nrho = nderiv == 0 ? 1 : (nderiv == 1 ? 4 : 5);
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = MAX(1, _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc));
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *aobuf = malloc(sizeof(double) * max_nao * (GEMM_NGRIDS*(ncomp+1) + max_nao));
        double *cbuf = aobuf + max_nao * GEMM_NGRIDS * ncomp;
        double *dmbuf = cbuf + max_nao * GEMM_NGRIDS;
        double *work = malloc(sizeof(double) * EVAL_WORK);
        double rho1[GEMM_NGRIDS];
        double *prho, *pin, *pout, *pa;
        int igroup, blk0, blk1, ig0, ng, ns, s, s1, i, c, k, n;
#pragma omp for schedule(dynamic, 4)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        for (c = 0; c < nrho; c++) {
                                prho = rho + c * Ngrids + ig0;
                                for (k = 0; k < ng; k++) {
                                        prho[k] = 0;
                                }
                        }
                        continue;
                }
                n = 0;
                for (s = 0; s < g.nseg; s++) {
                for (i = g.seg_ao[s]; i < g.seg_ao[s] + g.seg_len[s]; i++, n++) {
                        pin = dm + i * Nao;
                        pout = dmbuf + n * (size_t)ns;
                        for (s1 = 0; s1 < g.nseg; s1++) {
                                for (k = 0; k < g.seg_len[s1]; k++) {
                                        pout[g.seg_off[s1]+k] = pin[g.seg_ao[s1]+k];
                                }
                        }
                } }
                _eval_group(aobuf, deriv, &g, ng, coords, ldc, blk0, rcache, roffs,
                            nbas, atm, bas, env, work);
                dgemm_(&TRANS_N, &TRANS_T, &ng, &ns, &ns, &D1, aobuf, &ng,
                       dmbuf, &ns, &D0, cbuf, &ng);
                _gemm_contract_buf(rho + ig0, aobuf, cbuf, ns, ng);
                if (nderiv == 0) {
                        continue;
                }
                for (c = 1; c < 4; c++) {
                        _contract_group(rho1, aobuf + c * (size_t)ns * ng, cbuf, &g,
                                        ao_loc, ng);
                        prho = rho + c * Ngrids + ig0;
                        for (k = 0; k < ng; k++) {
                                prho[k] = rho1[k] * 2;
                        }
                }
                if (nderiv == 1) {
                        continue;
                }
                prho = rho + 4 * Ngrids + ig0;
                for (k = 0; k < ng; k++) {
                        prho[k] = 0;
                }
                for (c = 1; c < 4; c++) {
                        pa = aobuf + c * (size_t)ns * ng;
                        dgemm_(&TRANS_N, &TRANS_T, &ng, &ns, &ns, &D1, pa, &ng,
                               dmbuf, &ns, &D0, cbuf, &ng);
                        _gemm_contract_buf(rho1, pa, cbuf, ns, ng);
                        for (k = 0; k < ng; k++) {
                                prho[k] += rho1[k] * .5;
                        }
                }
        }
        _gemm_group_free(&g);
        free(aobuf);
        free(work);
}
}

/* The XC matrix kernels: _dot_ao_ao_gemm of nr_numint_sparse.c with the AO values computed
 * in place. tau == 0: out[i,j] += sum_g ao[0,i,g] sum_c wv[c,g] ao[c,j,g], c < ncomp
 * (VXCdot_ao_aow_sparse_gemm). tau == 1: out[i,j] += sum_g sum_c ao[c,i,g] wv[g] ao[c,j,g],
 * c = 1,2,3 (VXCdot_ao_wv_ao_sparse_gemm, the meta-GGA tau term, symmetric). */
static void _fused_dot(double *out, double *wv, int ncomp, int tau,
                       int nao, int ngrids, int nbas,
                       uint8_t *screen_index, int *ao_loc, double *coords, int ldc,
                       int *atm, int *bas, double *env, double *rcache, int64_t *roffs)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const size_t nn = Nao * Nao;
        const int deriv = (tau || ncomp > 1) ? 1 : 0;
        const int naocomp = deriv == 0 ? 1 : 4;
        const int hermi = tau ? HERMITIAN : 0;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc);
        double **vprivs = malloc(sizeof(double *) * omp_get_max_threads());
        if (max_nao > 0) {
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const char UPLO = 'U';
        const double D0 = 0;
        const double D1 = 1;
        const double DM1 = -1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *vpriv = calloc(nn, sizeof(double));
        vprivs[omp_get_thread_num()] = vpriv;
        // AO comps, then the scaled AOs (or the dsyrk operand of 3 comps), then the product
        double *aobuf = malloc(sizeof(double) * max_nao * (GEMM_NGRIDS*(naocomp+3) + max_nao));
        double *ketbuf = aobuf + max_nao * GEMM_NGRIDS * naocomp;
        double *cbuf = ketbuf + max_nao * GEMM_NGRIDS * 3;
        double *work = malloc(sizeof(double) * EVAL_WORK);
        double sqrtw[GEMM_NGRIDS];
        int pidx[GEMM_NGRIDS];
        int igroup, blk0, blk1, ig0, ng, ns, np, nk, sign, i, j, k, k0, k1, b, s, c, p, n, mask;
        size_t comp_size;
        double w, beta;
        double *pbuf, *pa, *pw;
#pragma omp for schedule(static, 1)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        continue;
                }
                comp_size = (size_t)ns * ng;
                _eval_group(aobuf, deriv, &g, ng, coords, ldc, blk0, rcache, roffs,
                            nbas, atm, bas, env, work);
                if (!tau) {
                        // ketbuf[n,k] = sum_c ao[c,n,k] wv[c,ig0+k], zero where screened
                        // (_gemm_gather with weights)
                        pbuf = ketbuf;
                        n = 0;
                        for (s = 0; s < g.nshl; s++) {
                        for (i = ao_loc[g.shl[s]]; i < ao_loc[g.shl[s]+1]; i++, n++) {
                                pa = aobuf + n * (size_t)ng;
                                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                                        k1 = MIN(k0 + BLKSIZE, ng);
                                        if (!(g.blkmask[s] & (1 << b))) {
                                                for (k = k0; k < k1; k++) {
                                                        pbuf[k] = 0;
                                                }
                                                continue;
                                        }
                                        pw = wv + ig0;
                                        for (k = k0; k < k1; k++) {
                                                pbuf[k] = pa[k] * pw[k];
                                        }
                                        for (c = 1; c < ncomp; c++) {
                                                pw = wv + c * Ngrids + ig0;
                                                for (k = k0; k < k1; k++) {
                                                        pbuf[k] += pa[c*comp_size+k] * pw[k];
                                                }
                                        }
                                }
                                pbuf += ng;
                        } }
                        dgemm_(&TRANS_T, &TRANS_N, &ns, &ns, &ng, &D1, ketbuf, &ng,
                               aobuf, &ng, &D0, cbuf, &ns);
                } else {
                        beta = 0;
                        for (sign = 1; sign >= -1; sign -= 2) {
                                np = 0;
                                for (k = 0; k < ng; k++) {
                                        w = wv[ig0+k] * sign;
                                        if (w > 0 || (sign > 0 && w != w)) {
                                                pidx[np] = k;
                                                sqrtw[np] = sqrt(w);
                                                np++;
                                        }
                                }
                                if (np == 0) {
                                        continue;
                                }
                                // ketbuf[n,c*np+p] = ao[1+c,n,pidx[p]] sqrtw[p], zero where
                                // screened (_gemm_gather_points)
                                pbuf = ketbuf;
                                n = 0;
                                for (s = 0; s < g.nshl; s++) {
                                        mask = g.blkmask[s];
                                        for (i = ao_loc[g.shl[s]]; i < ao_loc[g.shl[s]+1]; i++, n++) {
                                        for (c = 0; c < 3; c++) {
                                                pa = aobuf + (1+c) * comp_size + n * (size_t)ng;
                                                for (p = 0; p < np; p++) {
                                                        if (mask & (1 << (pidx[p] / BLKSIZE))) {
                                                                pbuf[p] = pa[pidx[p]] * sqrtw[p];
                                                        } else {
                                                                pbuf[p] = 0;
                                                        }
                                                }
                                                pbuf += np;
                                        } }
                                }
                                nk = np * 3;
                                dsyrk_(&UPLO, &TRANS_T, &ns, &nk, (sign > 0 ? &D1 : &DM1),
                                       ketbuf, &nk, &beta, cbuf, &ns);
                                beta = 1;
                        }
                        if (beta == 0) {
                                continue;
                        }
                }
                _gemm_scatter_add(vpriv, cbuf, &g, Nao, hermi);
        }
        NPomp_dsum_reduce_inplace(vprivs, nn);
        double *vsum = vprivs[0];
#pragma omp for schedule(static)
        for (i = 0; i < nao; i++) {
                for (j = 0; j < (hermi ? i+1 : nao); j++) {
                        out[i*Nao+j] += vsum[i*Nao+j];
                }
        }
        _gemm_group_free(&g);
        free(vpriv);
        free(aobuf);
        free(work);
}
        }
        free(vprivs);
        if (hermi != 0) {
                NPdsymm_triu(nao, out, hermi);
        }
}

void FUSEDdot_ao_aow(double *out, double *wv, int ncomp, int nao, int ngrids, int nbas,
                     uint8_t *screen_index, int *ao_loc, double *coords, int ldc,
                     int *atm, int *bas, double *env, double *rcache, int64_t *roffs)
{
        _fused_dot(out, wv, ncomp, 0, nao, ngrids, nbas, screen_index, ao_loc,
                   coords, ldc, atm, bas, env, rcache, roffs);
}

void FUSEDdot_ao_wv_ao(double *out, double *wv, int nao, int ngrids, int nbas,
                       uint8_t *screen_index, int *ao_loc, double *coords, int ldc,
                       int *atm, int *bas, double *env, double *rcache, int64_t *roffs)
{
        _fused_dot(out, wv, 3, 1, nao, ngrids, nbas, screen_index, ao_loc,
                   coords, ldc, atm, bas, env, rcache, roffs);
}
