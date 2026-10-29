/*
 * Integral time per fine class, and how much of it a semi-direct ERI cache of a given size holds
 * when it keeps (q) the quartets with the largest Schwarz products, (v) those with the largest
 * evaluation time per byte, or (vu) the largest time per byte times u(q), the share of the SCF
 * cycles expected to need the quartet (1 above q = 1e-7, falling linearly in log q to 0.05 at the
 * Schwarz cutoff 1e-13). Each policy is scored by the time it saves per cycle, sum(t u).
 *
 *   qcost LIB.so MOL.bin [--samples N] [--budget GB]... [--dump CLASSES.txt]
 *
 * Fine classes: angular momenta and primitive numbers of the four shells (s8-unique quartets with
 * q_ij q_kl > 1e-13, as in qbench). The class time is the mean over its first N quartets in loop
 * order (default 16), each the best of 3 timings of 16 calls. Single thread.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int (*FINT)(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas,
                    double *env, void *opt, double *cache);
typedef void (*FOPT)(void **opt, int *atm, int natm, int *bas, int nbas, double *env);

#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define BAS(slot, i) bas[(i) * 8 + (slot)]
#define NBIN 1000       // 0.02 decades from 1e-15
#define CUTOFF 1e-13

typedef struct {
        int natm, nbas, nenv;
        int *atm, *bas;
        double *env, *q;
} Mol;

typedef struct {
        uint64_t key;
        long n, nint;
        int ns;
        int *sh;        // samples, 4 per quartet
        double t;       // mean seconds per quartet
} Class;

static Mol load(const char *path)
{
        Mol m;
        FILE *f = fopen(path, "rb");
        int hdr[3];
        if (f == NULL || fread(hdr, sizeof(int), 3, f) != 3) exit(1);
        m.natm = hdr[0]; m.nbas = hdr[1]; m.nenv = hdr[2];
        m.atm = malloc(sizeof(int) * m.natm * 6);
        m.bas = malloc(sizeof(int) * m.nbas * 8);
        m.env = malloc(sizeof(double) * m.nenv);
        m.q = malloc(sizeof(double) * m.nbas * m.nbas);
        if (fread(m.atm, sizeof(int), m.natm * 6, f) != (size_t)m.natm * 6 ||
            fread(m.bas, sizeof(int), m.nbas * 8, f) != (size_t)m.nbas * 8 ||
            fread(m.env, sizeof(double), m.nenv, f) != (size_t)m.nenv ||
            fread(m.q, sizeof(double), (size_t)m.nbas * m.nbas, f) != (size_t)m.nbas * m.nbas) exit(1);
        fclose(f);
        return m;
}

static double now(void)
{
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return t.tv_sec + 1e-9 * t.tv_nsec;
}

#define HSIZE (1 << 16)
static Class *table[HSIZE];
static Class classes[HSIZE];
static int nclass = 0;

static Class *lookup(uint64_t key)
{
        uint64_t h = (key * 0x9E3779B97F4A7C15ull) >> 48;
        while (table[h] != NULL && table[h]->key != key) h = (h + 1) & (HSIZE - 1);
        if (table[h] == NULL) {
                Class *c = classes + nclass++;
                memset(c, 0, sizeof(Class));
                c->key = key;
                table[h] = c;
        }
        return table[h];
}

static uint64_t key_of(int *bas, int *sh)
{
        uint64_t k = 0;
        int n;
        for (n = 0; n < 4; n++) {
                k = (k << 4) | BAS(ANG_OF, sh[n]);
                k = (k << 8) | BAS(NPRIM_OF, sh[n]);
        }
        return k;
}

static int nf(int *bas, int sh)
{
        int l = BAS(ANG_OF, sh);
        return (2 * l + 1) * BAS(NCTR_OF, sh);
}

static int bin_of(double x)
{
        int b = (int)floor((log10(x) + 15) * 50);
        return b < 0 ? 0 : b >= NBIN ? NBIN - 1 : b;
}

static double use_of(double q)
{
        double u = (log10(q) - log10(CUTOFF)) / 6;
        return u > 1 ? 1 : u < .05 ? .05 : u;
}

int main(int argc, char **argv)
{
        if (argc < 3) {
                fprintf(stderr, "usage: qcost LIB.so MOL.bin [--samples N] [--budget GB]...\n");
                return 1;
        }
        Mol m = load(argv[2]);
        int nsamp = 16, nbudget = 0, a;
        double budgets[16];
        const char *dumpfile = NULL;
        for (a = 3; a < argc; a++) {
                if (!strcmp(argv[a], "--samples")) nsamp = atoi(argv[++a]);
                else if (!strcmp(argv[a], "--budget")) budgets[nbudget++] = atof(argv[++a]) * 1e9;
                else if (!strcmp(argv[a], "--dump")) dumpfile = argv[++a];
        }
        if (nbudget == 0) {
                budgets[0] = 5e9; budgets[1] = 10e9; budgets[2] = 15e9; budgets[3] = 20e9;
                nbudget = 4;
        }
        void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (lib == NULL) { fprintf(stderr, "%s\n", dlerror()); return 1; }
        FINT fint = (FINT)dlsym(lib, "int2e_sph");
        FOPT fopt = (FOPT)dlsym(lib, "int2e_optimizer");
        void *opt = NULL;
        fopt(&opt, m.atm, m.natm, m.bas, m.nbas, m.env);
        int nbas = m.nbas, *bas = m.bas;
        double *q = m.q;
        int sh[4], i, j, k, l, s;

        // pass 1: classes, their sizes and samples
        long nq = 0;
        for (i = 0; i < nbas; i++)
        for (j = 0; j <= i; j++)
        for (k = 0; k <= i; k++)
        for (l = 0; l <= k; l++) {
                if (k == i && l > j) break;
                if (q[i * nbas + j] * q[k * nbas + l] <= CUTOFF) continue;
                sh[0] = i; sh[1] = j; sh[2] = k; sh[3] = l;
                Class *c = lookup(key_of(bas, sh));
                long ni = 1;
                for (s = 0; s < 4; s++) ni *= nf(bas, sh[s]);
                c->n++;
                c->nint += ni;
                if (c->ns < nsamp) {
                        c->sh = realloc(c->sh, sizeof(int) * 4 * (c->ns + 1));
                        memcpy(c->sh + 4 * c->ns, sh, sizeof(sh));
                        c->ns++;
                }
                nq++;
        }
        // time the samples
        double *buf = malloc(sizeof(double) * (1 << 20));
        double *cache = malloc(sizeof(double) * (1 << 22));
        double ttot = 0;
        int ci, r, t;
        for (ci = 0; ci < nclass; ci++) {
                Class *c = classes + ci;
                double sum = 0;
                for (s = 0; s < c->ns; s++) {
                        double best = 1e30;
                        for (r = 0; r < 3; r++) {
                                double t0 = now();
                                for (t = 0; t < 16; t++)
                                        fint(buf, NULL, c->sh + 4 * s, m.atm, m.natm, bas, nbas, m.env, opt, cache);
                                double dt = (now() - t0) / 16;
                                if (dt < best) best = dt;
                        }
                        sum += best;
                }
                c->t = sum / c->ns;
                ttot += c->t * c->n;
        }
        // pass 2: histograms of bytes, time and time*use over log q, log(t/byte), log(t*u/byte)
        static double hq[NBIN][3], hv[NBIN][3], hvu[NBIN][3];
        double btot = 0, tutot = 0;
        for (i = 0; i < nbas; i++)
        for (j = 0; j <= i; j++)
        for (k = 0; k <= i; k++)
        for (l = 0; l <= k; l++) {
                if (k == i && l > j) break;
                double qq = q[i * nbas + j] * q[k * nbas + l];
                if (qq <= CUTOFF) continue;
                sh[0] = i; sh[1] = j; sh[2] = k; sh[3] = l;
                Class *c = lookup(key_of(bas, sh));
                double b = 8. * c->nint / c->n;
                double tq = c->t, u = use_of(qq);
                int bq = bin_of(qq), bv = bin_of(tq / b * 1e9), bvu = bin_of(tq * u / b * 1e9);
                hq[bq][0] += b; hq[bq][1] += tq; hq[bq][2] += tq * u;
                hv[bv][0] += b; hv[bv][1] += tq; hv[bv][2] += tq * u;
                hvu[bvu][0] += b; hvu[bvu][1] += tq; hvu[bvu][2] += tq * u;
                btot += b;
                tutot += tq * u;
        }
        printf("%s: %ld quartets, %d classes, %.2f GB, %.2f s per pass (1 thread, sampled)\n",
               argv[2], nq, nclass, btot / 1e9, ttot);
        printf("share of the time per cycle saved, sum(t u) (and of one full pass, sum(t)), by policy:\n");
        int bi;
        for (bi = 0; bi < nbudget; bi++) {
                double (*hs[3])[3] = {hq, hv, hvu};
                const char *names[3] = {"q", "v", "vu"};
                printf("  budget %5.1f GB:", budgets[bi] / 1e9);
                int p;
                for (p = 0; p < 3; p++) {
                        double bsum = 0, tsum = 0, tusum = 0;
                        int x;
                        for (x = NBIN - 1; x >= 0; x--) {
                                if (bsum + hs[p][x][0] > budgets[bi]) break;
                                bsum += hs[p][x][0]; tsum += hs[p][x][1]; tusum += hs[p][x][2];
                        }
                        printf("  %s %.3f (%.3f)", names[p], tusum / tutot, tsum / ttot);
                }
                printf("\n");
        }
        if (dumpfile != NULL) {
                // l(ijkl) nprim(ijkl) quartets integrals ns/quartet
                FILE *f = fopen(dumpfile, "w");
                for (ci = 0; ci < nclass; ci++) {
                        Class *c = classes + ci;
                        uint64_t key = c->key;
                        int ls[4], ps[4];
                        for (s = 3; s >= 0; s--) { ps[s] = key & 0xff; key >>= 8; ls[s] = key & 0xf; key >>= 4; }
                        fprintf(f, "%d %d %d %d %d %d %d %d %ld %ld %.2f\n", ls[0], ls[1], ls[2], ls[3],
                                ps[0], ps[1], ps[2], ps[3], c->n, c->nint, 1e9 * c->t);
                }
                fclose(f);
        }
        // the classes with the most time
        printf("classes by time: l(ijkl) nprim(ijkl) quartets time%% ns/quartet ns/integral\n");
        int *order = malloc(sizeof(int) * nclass);
        for (ci = 0; ci < nclass; ci++) order[ci] = ci;
        for (i = 0; i < nclass; i++)
        for (j = i + 1; j < nclass; j++) {
                if (classes[order[j]].t * classes[order[j]].n > classes[order[i]].t * classes[order[i]].n) {
                        int x = order[i]; order[i] = order[j]; order[j] = x;
                }
        }
        double cum = 0;
        for (i = 0; i < nclass && cum < .8 * ttot; i++) {
                Class *c = classes + order[i];
                uint64_t key = c->key;
                int ls[4], ps[4];
                for (s = 3; s >= 0; s--) { ps[s] = key & 0xff; key >>= 8; ls[s] = key & 0xf; key >>= 4; }
                printf("  %d%d%d%d %2d %2d %2d %2d %9ld %5.1f%% %8.0f %8.2f\n", ls[0], ls[1], ls[2], ls[3],
                       ps[0], ps[1], ps[2], ps[3], c->n, 100 * c->t * c->n / ttot, 1e9 * c->t,
                       1e9 * c->t * c->n / c->nint);
                cum += c->t * c->n;
        }
        return 0;
}
