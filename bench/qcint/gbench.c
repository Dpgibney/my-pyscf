/*
 * Time int1e_grids (the potential of AO pairs at grid points, the integral of PySCF's SGX exchange) of a libcint or
 * qcint build, single thread, on all shell pairs i >= j of a molecule and batches of SGX_BLKSIZE grid points, as
 * SGXnr_direct_k_drv calls it, by angular-momentum class (unscreened: SGX screens 10-60% of these pairs away).
 * For the same calls it also times
 *   - CINTrys_roots alone, once per primitive pair (eij below the cutoff) and grid point: the scalar root step;
 *   - int1e_grids on batches of 8 points: the cost per call, (t8 - 8 (t224 - t8) / 216) per call.
 *     gbench LIB.so MOL.bin GRID.bin [--batches N] [--compare LIB2.so] [--pairs same|diff]
 * With --compare it instead compares the integrals of the two libraries on the same calls: the fraction that is
 * bitwise equal and the largest difference, overall and by class.
 * MOL.bin is written by dump_basis.py (the q_cond block is ignored), GRID.bin by dump_sgx_grid.py (int32 ngrids,
 * float64 coords[ngrids*3]). --batches N times every k-th batch so that N batches are used (default: all).
 * --pairs same (diff) times only the shell pairs on one atom (on two atoms).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SGX_BLKSIZE 224
#define NGRIDS 11
#define PTR_GRIDS 12
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define PTR_EXP 5
#define PTR_COEFF 6
#define BAS_SLOTS 8
#define PTR_COORD 1
#define ATM_SLOTS 6
#define EXPCUTOFF 60.

typedef int (*FINT)(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas,
                    double *env, void *opt, double *cache);
typedef void (*FOPT)(void **opt, int *atm, int natm, int *bas, int nbas, double *env);
typedef int (*FROOTS)(int nroots, double x, double *u, double *w);

static double now(void)
{
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *xmalloc(size_t n)
{
        void *p = malloc(n);
        if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
        return p;
}

static void *sym(void *lib, const char *name)
{
        void *p = dlsym(lib, name);
        if (!p) { fprintf(stderr, "%s not in the library\n", name); exit(1); }
        return p;
}

int main(int argc, char **argv)
{
        if (argc < 4) {
                fprintf(stderr, "usage: gbench LIB.so MOL.bin GRID.bin [--batches N]\n");
                return 1;
        }
        int want_batches = 0;
        int pairs = 0;  /* --pairs same|diff: only shell pairs on one atom, or on two */
        const char *compare = NULL;
        for (int a = 4; a < argc - 1; a++) {
                if (!strcmp(argv[a], "--batches")) want_batches = atoi(argv[a + 1]);
                if (!strcmp(argv[a], "--pairs")) pairs = !strcmp(argv[a + 1], "same") ? 1 : 2;
                if (!strcmp(argv[a], "--compare")) compare = argv[a + 1];
        }
        FILE *f = fopen(argv[2], "rb");
        int hdr[3];
        if (!f || fread(hdr, sizeof(int), 3, f) != 3) { fprintf(stderr, "bad %s\n", argv[2]); return 1; }
        int natm = hdr[0], nbas = hdr[1], nenv = hdr[2];
        int *atm = xmalloc(sizeof(int) * natm * ATM_SLOTS);
        int *bas = xmalloc(sizeof(int) * nbas * BAS_SLOTS);
        if (fread(atm, sizeof(int), natm * ATM_SLOTS, f) != (size_t)natm * ATM_SLOTS ||
            fread(bas, sizeof(int), nbas * BAS_SLOTS, f) != (size_t)nbas * BAS_SLOTS) return 1;
        f = freopen(argv[3], "rb", f);
        int ngrids;
        if (!f || fread(&ngrids, sizeof(int), 1, f) != 1) { fprintf(stderr, "bad %s\n", argv[3]); return 1; }
        double *env = xmalloc(sizeof(double) * (nenv + (size_t)ngrids * 3));
        FILE *fm = fopen(argv[2], "rb");
        fseek(fm, sizeof(int) * (3 + natm * ATM_SLOTS + nbas * BAS_SLOTS), SEEK_SET);
        if (fread(env, sizeof(double), nenv, fm) != (size_t)nenv) return 1;
        fclose(fm);
        double *grids = env + nenv;
        if (fread(grids, sizeof(double), (size_t)ngrids * 3, f) != (size_t)ngrids * 3) return 1;
        fclose(f);
        env[NGRIDS] = ngrids;
        env[PTR_GRIDS] = nenv;

        void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
        FINT fint = (FINT)sym(lib, "int1e_grids_sph");
        FOPT fopt = (FOPT)sym(lib, "int1e_grids_optimizer");
        FROOTS froots = (FROOTS)sym(lib, "CINTrys_roots");
        void *opt = NULL;
        fopt(&opt, atm, natm, bas, nbas, env);

        int nbatch = (ngrids + SGX_BLKSIZE - 1) / SGX_BLKSIZE;
        int step = want_batches > 0 ? (nbatch + want_batches - 1) / want_batches : 1;
        if (step < 1) step = 1;
        if (compare) {
                /* the integrals of LIB.so against those of the --compare library, on the same batches */
                void *lib2 = dlopen(compare, RTLD_NOW | RTLD_LOCAL);
                if (!lib2) { fprintf(stderr, "%s\n", dlerror()); return 1; }
                FINT fint2 = (FINT)sym(lib2, "int1e_grids_sph");
                FOPT fopt2 = (FOPT)sym(lib2, "int1e_grids_optimizer");
                void *opt2 = NULL;
                fopt2(&opt2, atm, natm, bas, nbas, env);
                double *c1 = xmalloc(sizeof(double) * 4 << 20), *c2 = xmalloc(sizeof(double) * 4 << 20);
                double *b1 = xmalloc(sizeof(double) * 64 * 64 * SGX_BLKSIZE);
                double *b2 = xmalloc(sizeof(double) * 64 * 64 * SGX_BLKSIZE);
                double maxabs = 0, maxrel = 0, maxval = 0, nval = 0;
                long nbitwise = 0;
                int lmax2 = 0;
                for (int i = 0; i < nbas; i++) if (bas[i*BAS_SLOTS+ANG_OF] > lmax2) lmax2 = bas[i*BAS_SLOTS+ANG_OF];
                double *cval = calloc((lmax2 + 1) * (lmax2 + 1), sizeof(double));
                double *cbit = calloc((lmax2 + 1) * (lmax2 + 1), sizeof(double));
                double *cmax = calloc((lmax2 + 1) * (lmax2 + 1), sizeof(double));
                for (int b = 0; b < nbatch; b += step) {
                        int ig0 = b * SGX_BLKSIZE;
                        int ig1 = ig0 + SGX_BLKSIZE < ngrids ? ig0 + SGX_BLKSIZE : ngrids;
                        for (int i = 0; i < nbas; i++) {
                        for (int j = 0; j <= i; j++) {
                                int di = (2 * bas[i*BAS_SLOTS+ANG_OF] + 1) * bas[i*BAS_SLOTS+NCTR_OF];
                                int dj = (2 * bas[j*BAS_SLOTS+ANG_OF] + 1) * bas[j*BAS_SLOTS+NCTR_OF];
                                int shls[4] = {i, j, ig0, ig1};
                                int dims[3] = {di, dj, ig1 - ig0};
                                fint(b1, dims, shls, atm, natm, bas, nbas, env, opt, c1);
                                fint2(b2, dims, shls, atm, natm, bas, nbas, env, opt2, c2);
                                size_t n = (size_t)di * dj * (ig1 - ig0);
                                int li = bas[i*BAS_SLOTS+ANG_OF], lj = bas[j*BAS_SLOTS+ANG_OF];
                                int cls = (li > lj ? li : lj) * (lmax2 + 1) + (li > lj ? lj : li);
                                double pmax = 0;
                                for (size_t k = 0; k < n; k++) if (fabs(b2[k]) > pmax) pmax = fabs(b2[k]);
                                for (size_t k = 0; k < n; k++) {
                                        double d = fabs(b1[k] - b2[k]);
                                        nbitwise += b1[k] == b2[k];
                                        cbit[cls] += b1[k] == b2[k];
                                        if (d > cmax[cls]) cmax[cls] = d;
                                        if (d > maxabs) maxabs = d;
                                        if (pmax > 0 && d / pmax > maxrel) maxrel = d / pmax;
                                        if (fabs(b2[k]) > maxval) maxval = fabs(b2[k]);
                                }
                                nval += n;
                                cval[cls] += n;
                        } }
                }
                printf("%s against %s: %.0f integrals, %.4f%% bitwise equal, max |diff| %.2e (max |value| %.2e), "
                       "max |diff| / max |value| of its pair %.2e\n", argv[1], compare, nval, 100. * nbitwise / nval,
                       maxabs, maxval, maxrel);
                const char *L2 = "spdfgh";
                printf("bitwise equal, max |diff| by class:");
                for (int ka = 0; ka <= lmax2; ka++) {
                for (int kb = 0; kb <= ka; kb++) {
                        int c = ka * (lmax2 + 1) + kb;
                        if (cval[c] > 0) printf(" (%c%c) %.2f%% %.1e", L2[ka], L2[kb], 100 * cbit[c] / cval[c], cmax[c]);
                } }
                printf("\n");
                return 0;
        }
        int lmax = 0;
        for (int i = 0; i < nbas; i++) if (bas[i*BAS_SLOTS+ANG_OF] > lmax) lmax = bas[i*BAS_SLOTS+ANG_OF];
        int ncls = (lmax + 1) * (lmax + 1);
        double *t224 = calloc(ncls, sizeof(double)), *t8 = calloc(ncls, sizeof(double));
        double *troot = calloc(ncls, sizeof(double)), *nint = calloc(ncls, sizeof(double));
        double *ncall = calloc(ncls, sizeof(double)), *nroot = calloc(ncls, sizeof(double));
        double *ncall8 = calloc(ncls, sizeof(double));
        double *xhist = calloc(ncls * 3, sizeof(double));  /* x < 1, < 35+5n (libcint's asymptotic limit), above */
        size_t cache_size = 4 << 20;
        double *cache = xmalloc(sizeof(double) * cache_size);
        double *buf = xmalloc(sizeof(double) * 64 * 64 * SGX_BLKSIZE);
        double u[32], w[32], sink = 0;
        int used = 0;
        for (int b = 0; b < nbatch; b += step, used++) {
                int ig0 = b * SGX_BLKSIZE;
                int ig1 = ig0 + SGX_BLKSIZE < ngrids ? ig0 + SGX_BLKSIZE : ngrids;
                for (int i = 0; i < nbas; i++) {
                for (int j = 0; j <= i; j++) {
                        int same = bas[i*BAS_SLOTS+ATOM_OF] == bas[j*BAS_SLOTS+ATOM_OF];
                        if ((pairs == 1 && !same) || (pairs == 2 && same)) continue;
                        int li = bas[i*BAS_SLOTS+ANG_OF], lj = bas[j*BAS_SLOTS+ANG_OF];
                        int ka = li > lj ? li : lj, kb = li > lj ? lj : li;
                        int cls = ka * (lmax + 1) + kb;
                        int di = (2 * li + 1) * bas[i*BAS_SLOTS+NCTR_OF];
                        int dj = (2 * lj + 1) * bas[j*BAS_SLOTS+NCTR_OF];
                        int shls[4] = {i, j, ig0, ig1};
                        int dims[3] = {di, dj, ig1 - ig0};
                        double t = now();
                        fint(buf, dims, shls, atm, natm, bas, nbas, env, opt, cache);
                        t224[cls] += now() - t;
                        sink += buf[0];
                        ncall[cls] += 1;
                        nint[cls] += (double)di * dj * (ig1 - ig0);
                        if (ig1 - ig0 >= 8) {
                                int shls8[4] = {i, j, ig0, ig0 + 8};
                                int dims8[3] = {di, dj, 8};
                                t = now();
                                fint(buf, dims8, shls8, atm, natm, bas, nbas, env, opt, cache);
                                t8[cls] += now() - t;
                                ncall8[cls] += 1;
                                sink += buf[0];
                        }
                        /* the roots of the same primitive pairs and points */
                        int nroots = (li + lj) / 2 + 1;
                        double *ri = env + atm[bas[i*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
                        double *rj = env + atm[bas[j*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
                        double rr = 0;
                        for (int x = 0; x < 3; x++) rr += (ri[x] - rj[x]) * (ri[x] - rj[x]);
                        double *ai = env + bas[i*BAS_SLOTS+PTR_EXP], *aj = env + bas[j*BAS_SLOTS+PTR_EXP];
                        int ni = bas[i*BAS_SLOTS+NPRIM_OF], nj = bas[j*BAS_SLOTS+NPRIM_OF];
                        t = now();
                        for (int jp = 0; jp < nj; jp++) {
                        for (int ip = 0; ip < ni; ip++) {
                                double aij = ai[ip] + aj[jp];
                                if (rr * ai[ip] * aj[jp] / aij > EXPCUTOFF) continue;
                                double p[3];
                                for (int x = 0; x < 3; x++) p[x] = (ai[ip] * ri[x] + aj[jp] * rj[x]) / aij;
                                for (int g = ig0; g < ig1; g++) {
                                        double *rg = grids + 3 * g;
                                        double d0 = rg[0] - p[0], d1 = rg[1] - p[1], d2 = rg[2] - p[2];
                                        double xx = aij * (d0 * d0 + d1 * d1 + d2 * d2);
                                        froots(nroots, xx, u, w);
                                        sink += u[0] + w[0];
                                        xhist[cls * 3 + (xx < 1 ? 0 : xx < 35 + 5 * nroots ? 1 : 2)] += 1;
                                }
                                nroot[cls] += ig1 - ig0;
                        } }
                        troot[cls] += now() - t;
                } }
        }
        printf("%s: %d grid points, %d of %d batches of %d, nbas %d (single thread)\n",
               argv[1], ngrids, used, nbatch, SGX_BLKSIZE, nbas);
        printf("%-6s %10s %10s %9s %8s %10s %9s %10s %8s\n", "class", "calls", "integrals", "time s",
               "ns/int", "roots/int", "roots s", "roots %", "call ns");
        double T = 0, TR = 0, NI = 0, TC = 0;
        const char *L = "spdfgh";
        for (int ka = 0; ka <= lmax; ka++) {
        for (int kb = 0; kb <= ka; kb++) {
                int c = ka * (lmax + 1) + kb;
                if (ncall[c] == 0) continue;
                /* per-call cost from the 224- and 8-point timings */
                double per224 = t224[c] / ncall[c], per8 = t8[c] / ncall8[c];
                double fixed = per8 - 8 * (per224 - per8) / (SGX_BLKSIZE - 8);
                printf("(%c%c)  %10.0f %10.3e %9.3f %8.2f %10.2f %9.3f %9.1f%% %8.0f\n", L[ka], L[kb], ncall[c],
                       nint[c], t224[c], 1e9 * t224[c] / nint[c], nroot[c] / nint[c], troot[c],
                       100 * troot[c] / t224[c], 1e9 * fixed);
                T += t224[c]; TR += troot[c]; NI += nint[c]; TC += fixed * ncall[c];
        } }
        printf("all    %10s %10.3e %9.3f %8.2f %10s %9.3f %9.1f%% (per-call cost %.1f%% of the time)\n", "", NI, T,
               1e9 * T / NI, "", TR, 100 * TR / T, 100 * TC / T);
        printf("root evaluations by x: class, x < 1, 1 <= x < 35+5 nroots, asymptotic\n");
        for (int ka = 0; ka <= lmax; ka++) {
        for (int kb = 0; kb <= ka; kb++) {
                int c = ka * (lmax + 1) + kb;
                double tot = xhist[3*c] + xhist[3*c+1] + xhist[3*c+2];
                if (tot == 0) continue;
                printf("(%c%c)  %6.1f%% %6.1f%% %6.1f%%\n", L[ka], L[kb], 100 * xhist[3*c] / tot,
                       100 * xhist[3*c+1] / tot, 100 * xhist[3*c+2] / tot);
        } }
        if (sink == 12345.678) printf("%g\n", sink);
        return 0;
}
