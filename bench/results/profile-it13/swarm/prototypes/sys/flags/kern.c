// Verbatim copies of hot non-BLAS helpers of pyscf/lib/dft/nr_numint_fused.c and nr_numint_gemm.h (perf d88076cb8)
#include <stdlib.h>
#include <stdint.h>
#define BLKSIZE 56
#define GEMM_NBLK 4
#define GEMM_NGRIDS (GEMM_NBLK*BLKSIZE)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
typedef struct { int nao, nshl, nseg; int *shl; uint8_t *blkmask; int *seg_ao, *seg_off, *seg_len; } GemmGroup;

void k_gemm_scatter_add(double *out, double *cbuf, GemmGroup *g, size_t Nao, int tril)
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
                                if (m0 > n) break;
                                mlen = MIN(mlen, n - m0 + 1);
                        }
                        for (m = 0; m < mlen; m++) {
                                pout[g->seg_ao[s1]+m] += pc[m0+m];
                        }
                }
        } }
}

void k_gemm_contract_buf(double *rho, double *a, double *c, int ns, int ng)
{
        int n, k;
        double *pa, *pc;
        for (k = 0; k < ng; k++) rho[k] = 0;
        for (n = 0; n < ns; n++) {
                pa = a + n * (size_t)ng;
                pc = c + n * (size_t)ng;
                for (k = 0; k < ng; k++) rho[k] += pa[k] * pc[k];
        }
}

void k_contract_group(double *rho, double *a, double *c, GemmGroup *g, int *ao_loc, int ng)
{
        int s, i, b, k, k0, k1;
        int n = 0;
        double *pa, *pc;
        for (k = 0; k < ng; k++) rho[k] = 0;
        for (s = 0; s < g->nshl; s++) {
        for (i = ao_loc[g->shl[s]]; i < ao_loc[g->shl[s]+1]; i++, n++) {
                pa = a + n * (size_t)ng;
                pc = c + n * (size_t)ng;
                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                        if (g->blkmask[s] & (1 << b)) {
                                k1 = MIN(k0 + BLKSIZE, ng);
                                for (k = k0; k < k1; k++) rho[k] += pa[k] * pc[k];
                        }
                }
        } }
}

void k_contract_mo(double *rho, double *a, double *b, int nocc, int ng)
{
        int o, k;
        for (k = 0; k < ng; k++) rho[k] = a[k] * b[k];
        for (o = 1; o < nocc; o++)
                for (k = 0; k < ng; k++) rho[k] += a[o*(size_t)ng+k] * b[o*(size_t)ng+k];
}

// the GGA weighted gather of VXCfused_dot_wv (nr_numint_fused.c:593-618)
void k_gga_gather(double *opbuf, double *aobuf, double *wvi, size_t Ngrids, int ig0, int nwv,
                  GemmGroup *g, int *ao_loc, int ng)
{
        size_t comp_size = (size_t)g->nao * ng;
        double *pbuf = opbuf, *pa, *pw;
        int n = 0, s, j, b, k0, k1, k, c;
        for (s = 0; s < g->nshl; s++) {
        for (j = ao_loc[g->shl[s]]; j < ao_loc[g->shl[s]+1]; j++, n++) {
                pa = aobuf + n * (size_t)ng;
                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                        k1 = MIN(k0 + BLKSIZE, ng);
                        if (!(g->blkmask[s] & (1 << b))) {
                                for (k = k0; k < k1; k++) pbuf[k] = 0;
                                continue;
                        }
                        pw = wvi + ig0;
                        for (k = k0; k < k1; k++) pbuf[k] = pa[k] * pw[k];
                        for (c = 1; c < MIN(nwv, 4); c++) {
                                pw = wvi + c * Ngrids + ig0;
                                for (k = k0; k < k1; k++) pbuf[k] += pa[c*comp_size+k] * pw[k];
                        }
                }
                pbuf += ng;
        } }
}
