/*
 * The per-call setup of qcint's CINT1e_grids_loop, which an int1e_grids optimizer could precompute, on all shell
 * pairs i >= j of a molecule (20 passes): CINTOpt_log_max_pgto_coeff of both shells, CINTset_pairdata (an exp per
 * primitive pair), CINTOpt_non0coeff_byshell of both shells, in ns per call (each includes ~20 ns of timer), against
 * int1e_grids_sph on 8 grid points.
 *     callcost LIBCINT.so MOL.bin        (MOL.bin from dump_basis.py)
 *     gcc -O2 callcost.c -o callcost -ldl -lm
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define PTR_EXP 5
#define PTR_COEFF 6
#define BAS_SLOTS 8
#define PTR_COORD 1
#define ATM_SLOTS 6
typedef struct { double rij[3]; double eij; double cceij; } PairData;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv)
{
        void *lib = dlopen(argv[1], RTLD_NOW);
        void (*logmax)(double *, double *, int, int) = dlsym(lib, "CINTOpt_log_max_pgto_coeff");
        int (*pairdata)(PairData *, double *, double *, double *, double *, double *, double *, int, int, int, int,
                        double, double, double *) = dlsym(lib, "CINTset_pairdata");
        void (*non0)(int *, int *, double *, int, int) = dlsym(lib, "CINTOpt_non0coeff_byshell");
        int (*fint)(double *, int *, int *, int *, int, int *, int, double *, void *, double *) = dlsym(lib, "int1e_grids_sph");
        FILE *f = fopen(argv[2], "rb");
        int hdr[3];
        if (fread(hdr, sizeof(int), 3, f) != 3) return 1;
        int natm = hdr[0], nbas = hdr[1], nenv = hdr[2];
        int *atm = malloc(sizeof(int) * natm * ATM_SLOTS), *bas = malloc(sizeof(int) * nbas * BAS_SLOTS);
        if (fread(atm, sizeof(int), natm * ATM_SLOTS, f) != (size_t)natm * ATM_SLOTS) return 1;
        if (fread(bas, sizeof(int), nbas * BAS_SLOTS, f) != (size_t)nbas * BAS_SLOTS) return 1;
        double *env = malloc(sizeof(double) * (nenv + 64 * 3));
        if (fread(env, sizeof(double), nenv, f) != (size_t)nenv) return 1;
        double *grids = env + nenv;
        for (int g = 0; g < 64 * 3; g++) grids[g] = 3.0 * sin(g * 1.7);
        env[11] = 64; env[12] = nenv;
        double lmc[64], buf[64 * 64 * 64]; PairData pd[1024]; int idx[2048], cnt[64];
        double *cache = malloc(sizeof(double) << 22);
        double t0, tl = 0, tp = 0, tn = 0, tc = 0; long n = 0; double sink = 0;
        for (int rep = 0; rep < 20; rep++)
        for (int i = 0; i < nbas; i++) for (int j = 0; j <= i; j++) {
                int ni = bas[i*BAS_SLOTS+NPRIM_OF], nj = bas[j*BAS_SLOTS+NPRIM_OF];
                int ci = bas[i*BAS_SLOTS+NCTR_OF], cj = bas[j*BAS_SLOTS+NCTR_OF];
                double *ai = env + bas[i*BAS_SLOTS+PTR_EXP], *aj = env + bas[j*BAS_SLOTS+PTR_EXP];
                double *cfi = env + bas[i*BAS_SLOTS+PTR_COEFF], *cfj = env + bas[j*BAS_SLOTS+PTR_COEFF];
                double *ri = env + atm[bas[i*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
                double *rj = env + atm[bas[j*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
                double rr = (ri[0]-rj[0])*(ri[0]-rj[0]) + (ri[1]-rj[1])*(ri[1]-rj[1]) + (ri[2]-rj[2])*(ri[2]-rj[2]);
                t0 = now(); logmax(lmc, cfi, ni, ci); logmax(lmc + ni, cfj, nj, cj); tl += now() - t0;
                t0 = now(); pairdata(pd, ai, aj, ri, rj, lmc, lmc + ni, bas[i*BAS_SLOTS+ANG_OF], bas[j*BAS_SLOTS+ANG_OF], ni, nj, rr, 60., env); tp += now() - t0;
                t0 = now(); non0(idx, cnt, cfi, ni, ci); non0(idx + 1024, cnt + 32, cfj, nj, cj); tn += now() - t0;
                int shls[4] = {i, j, 0, 8};
                int dims[3] = {(2*bas[i*BAS_SLOTS+ANG_OF]+1)*ci, (2*bas[j*BAS_SLOTS+ANG_OF]+1)*cj, 8};
                t0 = now(); fint(buf, dims, shls, atm, natm, bas, nbas, env, NULL, cache); tc += now() - t0;
                sink += pd[0].eij + lmc[0] + cnt[0] + buf[0];
                n++;
        }
        printf("%s, %s: ns per call: log_maxc %.0f, pairdata %.0f, non0coeff %.0f; int1e_grids on 8 points %.0f%s\n",
               argv[1], argv[2], 1e9 * tl / n, 1e9 * tp / n, 1e9 * tn / n, 1e9 * tc / n, sink == 1.2345 ? "!" : "");
        return 0;
}
