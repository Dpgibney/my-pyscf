#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#define BLKSIZE 56
typedef struct { int nao, nshl, nseg; int *shl; uint8_t *blkmask; int *seg_ao, *seg_off, *seg_len; } GemmGroup;
void k_gemm_scatter_add(double *out, double *cbuf, GemmGroup *g, size_t Nao, int tril);
void k_gemm_contract_buf(double *rho, double *a, double *c, int ns, int ng);
void k_contract_group(double *rho, double *a, double *c, GemmGroup *g, int *ao_loc, int ng);
void k_contract_mo(double *rho, double *a, double *b, int nocc, int ng);
void k_gga_gather(double *opbuf, double *aobuf, double *wvi, size_t Ngrids, int ig0, int nwv, GemmGroup *g, int *ao_loc, int ng);
void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps, double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc, size_t nao, size_t ngrids, size_t blksize);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double chk(double *x, size_t n) { double s = 0; for (size_t i = 0; i < n; i++) s += x[i] * (1 + 1e-3 * (i % 7)); return s; }
#define TIME(label, body, reps) do { double best = 1e9; for (int b_ = 0; b_ < 5; b_++) { double t0 = now(); \
        for (int r_ = 0; r_ < reps; r_++) { body; } double t = (now() - t0) / reps; if (t < best) best = t; } \
        printf("%-44s %9.3f us\n", label, best * 1e6); } while (0)
int main(void) {
        const int ng = 224, ns = 160, nocc = 51, Nao = 246;
        // a group: 160 AOs from 60 shells (s,p,d mix) in 20 contiguous segments of the 246 AOs
        GemmGroup g; int shl[100], seg_ao[40], seg_off[40], seg_len[40]; uint8_t mask[100]; int ao_loc[101];
        ao_loc[0] = 0; for (int i = 0; i < 100; i++) { int l = (i % 5 == 4) ? 2 : (i % 3 == 2 ? 1 : 0); ao_loc[i+1] = ao_loc[i] + 2*l+1; }
        int nao_g = 0, nshl = 0, nseg = 0;
        for (int i = 0; i < 100 && nao_g < ns; i++) {
                if (i % 5 == 3) continue;          // a screened shell between segments
                if (nseg == 0 || seg_ao[nseg-1] + seg_len[nseg-1] != ao_loc[i]) { seg_ao[nseg] = ao_loc[i]; seg_off[nseg] = nao_g; seg_len[nseg] = 0; nseg++; }
                int d = ao_loc[i+1] - ao_loc[i]; seg_len[nseg-1] += d; shl[nshl] = i; mask[nshl] = (i % 11 == 0) ? 0x7 : 0xf; nshl++; nao_g += d;
        }
        g.nao = nao_g; g.nshl = nshl; g.nseg = nseg; g.shl = shl; g.blkmask = mask; g.seg_ao = seg_ao; g.seg_off = seg_off; g.seg_len = seg_len;
        int nsg = g.nao;
        double *ao = aligned_alloc(64, sizeof(double) * nsg * ng * 4 + 64), *c = aligned_alloc(64, sizeof(double) * nsg * ng + 64);
        double *op = aligned_alloc(64, sizeof(double) * nsg * ng + 64), *rho = aligned_alloc(64, sizeof(double) * ng * 2);
        double *wv = aligned_alloc(64, sizeof(double) * 4 * 67200), *cbuf = aligned_alloc(64, sizeof(double) * nsg * nsg);
        double *vmat = aligned_alloc(64, sizeof(double) * Nao * Nao);
        for (size_t i = 0; i < (size_t)nsg * ng * 4; i++) ao[i] = 1e-2 * (i % 101) - .4;
        for (size_t i = 0; i < (size_t)nsg * ng; i++) c[i] = 1e-2 * (i % 89) - .3;
        for (size_t i = 0; i < 4 * 67200; i++) wv[i] = 1e-3 * (i % 37) - .01;
        for (size_t i = 0; i < (size_t)nsg * nsg; i++) cbuf[i] = 1e-3 * (i % 53);
        memset(vmat, 0, sizeof(double) * Nao * Nao);
        printf("group ns=%d nshl=%d nseg=%d ng=%d\n", nsg, nshl, nseg, ng);
        TIME("_gemm_contract_buf (ns x 224)", k_gemm_contract_buf(rho, ao, c, nsg, ng), 2000);
        printf("   chk %.17g\n", chk(rho, ng));
        TIME("_contract_group (ns x 224, block masks)", k_contract_group(rho, ao + nsg*ng, c, &g, ao_loc, ng), 2000);
        printf("   chk %.17g\n", chk(rho, ng));
        TIME("_contract_mo (nocc=51 x 224)", k_contract_mo(rho, ao, c, nocc, ng), 5000);
        printf("   chk %.17g\n", chk(rho, ng));
        TIME("GGA weighted gather of VXCfused_dot_wv", k_gga_gather(op, ao, wv, 67200, 6720, 4, &g, ao_loc, ng), 2000);
        printf("   chk %.17g\n", chk(op, (size_t)nsg * ng));
        TIME("_gemm_scatter_add (ns x ns into 246^2)", k_gemm_scatter_add(vmat, cbuf, &g, Nao, 0), 2000);
        printf("   chk %.17g\n", chk(vmat, (size_t)Nao * Nao));
        // GTOshell_eval_grid_cart_deriv1 for a 56-point block: l = 0, 1, 2 shells, nc = 1, into a [4,ns,224] buffer
        double coord[3*BLKSIZE], exps[2*40*BLKSIZE], alpha[4] = {1.2, .5, .2, .1}, coeff[4] = {.5, .3, .2, .1}, env[8] = {0}, ri[3] = {0};
        for (int i = 0; i < 3*BLKSIZE; i++) coord[i] = .1 + 1e-2 * i;
        for (int i = 0; i < 2*40*BLKSIZE; i++) exps[i] = .3 + 1e-3 * (i % 57);
        double *gto = aligned_alloc(64, sizeof(double) * 4 * nsg * ng + 64), *cart = aligned_alloc(64, sizeof(double) * 4 * 6 * BLKSIZE + 64);
        TIME("deriv1 l=0 (1 AO, 56 points)", GTOshell_eval_grid_cart_deriv1(gto, ri, exps, coord, alpha, coeff, env, 0, 4, 1, nsg, ng, BLKSIZE), 200000);
        TIME("deriv1 l=1 (3 AOs, 56 points)", GTOshell_eval_grid_cart_deriv1(gto, ri, exps, coord, alpha, coeff, env, 1, 4, 1, nsg, ng, BLKSIZE), 100000);
        TIME("deriv1 l=2 (6 cart, 56 points)", GTOshell_eval_grid_cart_deriv1(cart, ri, exps, coord, alpha, coeff, env, 2, 4, 1, 6, BLKSIZE, BLKSIZE), 100000);
        printf("   chk %.17g %.17g\n", chk(gto, (size_t)4 * nsg * ng), chk(cart, 4 * 6 * BLKSIZE));
        return 0;
}
