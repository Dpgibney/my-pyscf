/*
 * qcint's batched int2e_sph (int2e_sph_batch: several quartets of one class in the SIMD lanes) against
 * int2e_sph quartet by quartet, on the s8-unique quartets with q_ij q_kl > 1e-13 of a dump_basis.py
 * molecule. Quartets are grouped by type (l and number of primitives of each shell, in quartet order);
 * int2e_sph_batch_ok says which types the batched code takes.
 *
 *   bbench LIB.so MOL.bin [--lr] [--ip1] [--sample N] [--reps N]
 *
 * --ip1: int2e_ip1_sph_batch against int2e_ip1_sph (3 components), on every ordered (i, j) with k >= l, as
 * the gradients take them (qbench's ip1).
 *
 * --sample N: up to N quartets of each type, evenly spaced, timed (best of --reps) and scaled by the
 * type's count to project one pass (default). --sample 0: every quartet, compared but not timed.
 * Every computed integral is compared.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BAS(slot, i) bas[(i) * 8 + (slot)]
#define ANG_OF 1
#define NPRIM_OF 2
#define PTR_RANGE_OMEGA 8

typedef int (*FINT)(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas,
                    double *env, void *opt, double *cache);
typedef int (*FBATCH)(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                      double *env, void *opt, double *cache);
typedef void (*FOPT)(void **opt, int *atm, int natm, int *bas, int nbas, double *env);

typedef struct {
        int natm, nbas, nenv;
        int *atm, *bas;
        double *env, *q;
} Mol;

static Mol load(const char *path)
{
        Mol m;
        FILE *f = fopen(path, "rb");
        if (f == NULL) {
                perror(path);
                exit(1);
        }
        int hdr[3];
        if (fread(hdr, sizeof(int), 3, f) != 3) exit(1);
        m.natm = hdr[0];
        m.nbas = hdr[1];
        m.nenv = hdr[2];
        m.atm = malloc(sizeof(int) * m.natm * 6);
        m.bas = malloc(sizeof(int) * m.nbas * 8);
        m.env = malloc(sizeof(double) * m.nenv);
        m.q = malloc(sizeof(double) * m.nbas * m.nbas);
        if (fread(m.atm, sizeof(int), m.natm * 6, f) != (size_t)m.natm * 6 ||
            fread(m.bas, sizeof(int), m.nbas * 8, f) != (size_t)m.nbas * 8 ||
            fread(m.env, sizeof(double), m.nenv, f) != (size_t)m.nenv ||
            fread(m.q, sizeof(double), (size_t)m.nbas * m.nbas, f) != (size_t)m.nbas * m.nbas) {
                fprintf(stderr, "short read %s\n", path);
                exit(1);
        }
        fclose(f);
        return m;
}

static double now(void)
{
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *sym(void *lib, const char *name)
{
        void *p = dlsym(lib, name);
        if (p == NULL) {
                fprintf(stderr, "%s\n", dlerror());
                exit(1);
        }
        return p;
}

typedef struct {
        long count;
        long seen;
        int n;
        int (*sh)[4];
} Type;

int main(int argc, char **argv)
{
        if (argc < 3) {
                fprintf(stderr, "usage: bbench LIB.so MOL.bin [--lr] [--ip1] [--sample N] [--reps N]\n");
                return 1;
        }
        Mol m = load(argv[2]);
        int reps = 2, nsample = 4000, ip1 = 0;
        for (int a = 3; a < argc; a++) {
                if (!strcmp(argv[a], "--reps")) reps = atoi(argv[++a]);
                else if (!strcmp(argv[a], "--sample")) nsample = atoi(argv[++a]);
                else if (!strcmp(argv[a], "--lr")) m.env[PTR_RANGE_OMEGA] = 0.3;
                else if (!strcmp(argv[a], "--ip1")) ip1 = 1;
        }
        const int ncomp = ip1 ? 3 : 1;
        void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (lib == NULL) {
                fprintf(stderr, "%s\n", dlerror());
                return 1;
        }
        FINT fint = (FINT)sym(lib, ip1 ? "int2e_ip1_sph" : "int2e_sph");
        FOPT fopt = (FOPT)sym(lib, ip1 ? "int2e_ip1_optimizer" : "int2e_optimizer");
        FBATCH fbatch = (FBATCH)sym(lib, ip1 ? "int2e_ip1_sph_batch" : "int2e_sph_batch");
        int (*fok)(int *, int *, double *) = sym(lib, ip1 ? "int2e_ip1_sph_batch_ok" : "int2e_sph_batch_ok");
        int lanes = ((int (*)())sym(lib, "int2e_sph_batch_lanes"))();
        size_t bcache = ((size_t (*)())sym(lib, "int2e_sph_batch_cache_size"))();
        void *opt = NULL;
        fopt(&opt, m.atm, m.natm, m.bas, m.nbas, m.env);

        int *bas = m.bas, nbas = m.nbas;
        int stype[8192], ntype = 0, tl[64], tp[64];
        for (int s = 0; s < nbas; s++) {
                int t;
                for (t = 0; t < ntype; t++)
                        if (tl[t] == BAS(ANG_OF, s) && tp[t] == BAS(NPRIM_OF, s)) break;
                if (t == ntype) {
                        tl[t] = BAS(ANG_OF, s);
                        tp[t] = BAS(NPRIM_OF, s);
                        ntype++;
                }
                stype[s] = t;
        }
        int nqt = ntype * ntype * ntype * ntype;
        Type *qt = calloc(nqt, sizeof(Type));
        long nq = 0;
        for (int pass = 0; pass < 2; pass++) {
                for (int i = 0; i < nbas; i++)
                for (int j = 0; j < (ip1 ? nbas : i + 1); j++)
                for (int k = 0; k < (ip1 ? nbas : i + 1); k++)
                for (int l = 0; l <= k; l++) {
                        if (!ip1 && k == i && l > j) break;
                        if (m.q[i * nbas + j] * m.q[k * nbas + l] <= 1e-13) continue;
                        Type *t = qt + ((stype[i] * ntype + stype[j]) * ntype + stype[k]) * ntype + stype[l];
                        if (pass == 0) {
                                t->count++;
                                nq++;
                                continue;
                        }
                        long cap = nsample > 0 && t->count > nsample ? nsample : t->count;
                        long stride = t->count > cap ? t->count / cap : 1;
                        if (t->seen++ % stride == 0 && t->n < cap) {
                                if (t->sh == NULL) t->sh = malloc(sizeof(int[4]) * cap);
                                int *s = t->sh[t->n++];
                                s[0] = i; s[1] = j; s[2] = k; s[3] = l;
                        }
                }
        }
        printf("%s%s: %ld quartets, %d shell types, %d lanes; %s\n", ip1 ? "int2e_ip1, " : "",
               m.env[PTR_RANGE_OMEGA] > 0 ? "long range (omega 0.3)" : "full range", nq, ntype, lanes,
               nsample > 0 ? "sampled per quartet type and timed" : "every quartet compared, not timed");

        // the largest block of integrals of a type's quartets
        size_t maxblk = 0;
        for (int t = 0; t < nqt; t++) {
                if (qt[t].n == 0) continue;
                size_t nf = (size_t)qt[t].n * ncomp;
                for (int x = 0; x < 4; x++) nf *= 2 * BAS(ANG_OF, qt[t].sh[0][x]) + 1;
                maxblk = nf > maxblk ? nf : maxblk;
        }
        double *ref = malloc(sizeof(double) * maxblk);
        double *bout = malloc(sizeof(double) * maxblk);
        double *cache = malloc(sizeof(double) * (1 << 22));
        double *bc = malloc(sizeof(double) * bcache);
        if (ref == NULL || bout == NULL || cache == NULL || bc == NULL) {
                fprintf(stderr, "malloc of %zu doubles failed\n", maxblk);
                return 1;
        }
        double tq_all = 0, tb_all = 0, dmax = 0, vmax = 0, rmax = 0;
        long nbatch = 0;
        double cat_q[3][3] = {{0}}, cat_b[3][3] = {{0}};  // [max l: <=1, 2, >=3][combinations: 1, 2-7, >=8]
        for (int t = 0; t < nqt; t++) {
                Type *T = qt + t;
                if (T->n == 0) continue;
                int *s0 = T->sh[0], lmax = 0;
                long ncomb = 1, nf = ncomp;
                for (int x = 0; x < 4; x++) {
                        int l = BAS(ANG_OF, s0[x]);
                        lmax = l > lmax ? l : lmax;
                        ncomb *= BAS(NPRIM_OF, s0[x]);
                        nf *= 2 * l + 1;
                }
                int ok = fok(s0, m.bas, m.env);
                double tq = 1e300, tb = 1e300;
                for (int rep = 0; rep < (nsample > 0 ? reps : 1); rep++) {
                        double t0 = now();
                        for (int n = 0; n < T->n; n++)
                                fint(ref + nf * n, NULL, T->sh[n], m.atm, m.natm, m.bas, m.nbas, m.env, opt, cache);
                        double t1 = now();
                        if (ok) {
                                fbatch(bout, T->n, &T->sh[0][0], m.atm, m.natm, m.bas, m.nbas, m.env, NULL, bc);
                        }
                        double t2 = now();
                        tq = t1 - t0 < tq ? t1 - t0 : tq;
                        tb = t2 - t1 < tb ? t2 - t1 : tb;
                }
                if (ok) {
                        nbatch += T->n;
                        for (long n = 0; n < T->n * nf; n++) {
                                double v = ref[n], d = fabs(v - bout[n]);
                                vmax = fmax(vmax, fabs(v));
                                dmax = fmax(dmax, d);
                                rmax = fmax(rmax, d / fmax(fabs(v), 1e-6));
                        }
                } else {
                        tb = tq;  // the batched path leaves these to int2e_sph
                }
                double scale = (double)T->count / T->n;
                int cl = lmax <= 1 ? 0 : lmax == 2 ? 1 : 2, cp = ncomb == 1 ? 0 : ncomb < 8 ? 1 : 2;
                cat_q[cl][cp] += tq * scale;
                cat_b[cl][cp] += tb * scale;
                tq_all += tq * scale;
                tb_all += tb * scale;
        }
        printf("batched: %ld of the %s quartets; max|d| %.2e (max|v| %.2e), max|d|/max(|v|,1e-6) %.2e\n",
               nbatch, nsample > 0 ? "sampled" : "", dmax, vmax, rmax);
        if (nsample > 0) {
                const char *lname[3] = {"max l <= 1", "max l = 2", "max l >= 3"};
                const char *pname[3] = {"1 prim. comb.", "2-7", ">= 8"};
                printf("%-12s %-14s %12s %12s %7s\n", "", "", "int2e_sph (s)", "batched (s)", "ratio");
                for (int cl = 0; cl < 3; cl++)
                        for (int cp = 0; cp < 3; cp++)
                                if (cat_q[cl][cp] > 0)
                                        printf("%-12s %-14s %12.1f %12.1f %6.2fx\n", lname[cl], pname[cp],
                                               cat_q[cl][cp], cat_b[cl][cp], cat_q[cl][cp] / cat_b[cl][cp]);
                printf("%-27s %12.1f %12.1f %6.2fx   (projected one pass, 1 thread; >= 8 combinations stay "
                       "on int2e_sph)\n", "all", tq_all, tb_all, tq_all / tb_all);
        }
        return 0;
}
