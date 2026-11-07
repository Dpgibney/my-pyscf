/*
 * qcint's batched int2e_sph (int2e_sph_batch: several quartets of one class in the SIMD lanes) against
 * int2e_sph quartet by quartet, on the s8-unique quartets with q_ij q_kl > 1e-13 of a dump_basis.py
 * molecule. Quartets are grouped by type (l and number of primitives of each shell, in quartet order);
 * int2e_sph_batch_ok says which types the batched code takes.
 *
 *   bbench LIB.so MOL.bin [--lr] [--int NAME] [--sample N] [--reps N]
 *
 * --int NAME: int2e (the default), or the derivatives ip1 (int2e_ip1_sph_batch against int2e_ip1_sph,
 * 3 components; --ip1 for short), ipip1, ipvip1 or ip1ip2 (Hessians, 9 components), on every ordered
 * (i, j) with k >= l, as the gradients take them (qbench's ip1).
 *
 * --dot (with ip1, ipip1, ipvip1 or ip1ip2): int2e_*_sph_batch_dot, the integrals contracted with 2
 * arrays of each quartet inside the kernels, against int2e_*_sph contracted here (both timed, the
 * batched integrals contracted here too); the arrays are random in [-1, 1).
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
typedef int (*FDOT)(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                    double *env, const double *gam, int ng, double *cache);

// out[(q*ng+n)*ncomp+c] = sum_f buf[(q*ncomp+c)*nf+f] gam[(q*ng+n)*nf+f]
static void contract_n(double *out, const double *buf, const double *gam, int nq, int ng, int ncomp,
                       long nf)
{
        for (int q = 0; q < nq; q++)
        for (int n = 0; n < ng; n++)
        for (int c = 0; c < ncomp; c++) {
                const double *b = buf + ((long)q * ncomp + c) * nf, *g = gam + ((long)q * ng + n) * nf;
                double sum = 0;
                for (long f = 0; f < nf; f++) sum += b[f] * g[f];
                out[((long)q * ng + n) * ncomp + c] = sum;
        }
}

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
                fprintf(stderr, "usage: bbench LIB.so MOL.bin [--lr] [--int NAME] [--sample N] [--reps N]\n");
                return 1;
        }
        Mol m = load(argv[2]);
        int reps = 2, nsample = 4000, dot = 0;
        const char *name = "int2e";
        for (int a = 3; a < argc; a++) {
                if (!strcmp(argv[a], "--reps")) reps = atoi(argv[++a]);
                else if (!strcmp(argv[a], "--dot")) dot = 1;
                else if (!strcmp(argv[a], "--sample")) nsample = atoi(argv[++a]);
                else if (!strcmp(argv[a], "--lr")) m.env[PTR_RANGE_OMEGA] = 0.3;
                else if (!strcmp(argv[a], "--ip1")) name = "ip1";
                else if (!strcmp(argv[a], "--int")) name = argv[++a];
        }
        // ip1: the gradients' integrals; ipip1, ipvip1, ip1ip2: the Hessians'
        const int ip1 = strcmp(name, "int2e") != 0;
        const int hess = ip1 && strcmp(name, "ip1") != 0;
        const int ncomp = hess ? 9 : ip1 ? 3 : 1;
        const int ng = 2;
        if (dot && !ip1) {
                fprintf(stderr, "--dot needs --int ip1, ipip1, ipvip1 or ip1ip2\n");
                return 1;
        }
        void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (lib == NULL) {
                fprintf(stderr, "%s\n", dlerror());
                return 1;
        }
        char fname[64];
        const char *pre = ip1 ? "int2e_" : "";
        snprintf(fname, sizeof(fname), "%s%s_sph", pre, name);
        FINT fint = (FINT)sym(lib, fname);
        snprintf(fname, sizeof(fname), "%s%s_optimizer", pre, name);
        FOPT fopt = (FOPT)sym(lib, fname);
        snprintf(fname, sizeof(fname), "%s%s_sph_batch", pre, name);
        FBATCH fbatch = (FBATCH)sym(lib, fname);
        snprintf(fname, sizeof(fname), "%s%s_sph_batch_ok", pre, name);
        int (*fok)(int *, int *, double *) = sym(lib, fname);
        FDOT fdot = NULL;
        if (dot) {
                snprintf(fname, sizeof(fname), "%s%s_sph_batch_dot", pre, name);
                fdot = (FDOT)sym(lib, fname);
        }
        int lanes = ((int (*)())sym(lib, "int2e_sph_batch_lanes"))();
        size_t bcache = ((size_t (*)())sym(lib, hess ? "int2e_hess_sph_batch_cache_size"
                                               : "int2e_sph_batch_cache_size"))();
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
        printf("%s%s%s%s: %ld quartets, %d shell types, %d lanes; %s\n", pre, ip1 ? name : "", ip1 ? ", " : "",
               m.env[PTR_RANGE_OMEGA] > 0 ? "long range (omega 0.3)" : "full range", nq, ntype, lanes,
               nsample > 0 ? "sampled per quartet type and timed" : "every quartet compared, not timed");

        // the largest block of integrals of a type's quartets
        size_t maxblk = 0, maxn = 0;
        for (int t = 0; t < nqt; t++) {
                if (qt[t].n == 0) continue;
                size_t nf = (size_t)qt[t].n * ncomp;
                for (int x = 0; x < 4; x++) nf *= 2 * BAS(ANG_OF, qt[t].sh[0][x]) + 1;
                maxblk = nf > maxblk ? nf : maxblk;
                maxn = (size_t)qt[t].n > maxn ? (size_t)qt[t].n : maxn;
        }
        double *ref = malloc(sizeof(double) * maxblk);
        double *bout = malloc(sizeof(double) * maxblk);
        // --dot: the arrays [quartet][ng][nf], and the contractions [quartet][ng][ncomp] (more than
        // the arrays when nf < ncomp)
        double *gam = malloc(sizeof(double) * (dot ? maxblk / ncomp * ng : 1));
        double *rdot = malloc(sizeof(double) * (dot ? maxn * ng * ncomp : 1));
        double *bdot = malloc(sizeof(double) * (dot ? maxn * ng * ncomp : 1));
        double *ddot = malloc(sizeof(double) * (dot ? maxn * ng * ncomp : 1));
        unsigned long long rng = 88172645463325252ull;
        double *cache = malloc(sizeof(double) * (1 << 22));
        double *bc = malloc(sizeof(double) * bcache);
        if (ref == NULL || bout == NULL || cache == NULL || bc == NULL) {
                fprintf(stderr, "malloc of %zu doubles failed\n", maxblk);
                return 1;
        }
        double tq_all = 0, tb_all = 0, dmax = 0, vmax = 0, rmax = 0;
        double td_all = 0, ddmax = 0, dvmax = 0;
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
                const long nfs = nf / ncomp;
                if (dot) {
                        for (long n = 0; n < (long)T->n * ng * nfs; n++) {
                                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                                gam[n] = (double)(rng >> 11) / (double)(1ull << 52) - 1;
                        }
                }
                double tq = 1e300, tb = 1e300, td = 1e300;
                for (int rep = 0; rep < (nsample > 0 ? reps : 1); rep++) {
                        double t0 = now();
                        for (int n = 0; n < T->n; n++)
                                fint(ref + nf * n, NULL, T->sh[n], m.atm, m.natm, m.bas, m.nbas, m.env, opt, cache);
                        if (dot) contract_n(rdot, ref, gam, T->n, ng, ncomp, nfs);
                        double t1 = now();
                        if (ok) {
                                fbatch(bout, T->n, &T->sh[0][0], m.atm, m.natm, m.bas, m.nbas, m.env, NULL, bc);
                                if (dot) contract_n(bdot, bout, gam, T->n, ng, ncomp, nfs);
                        }
                        double t2 = now();
                        if (ok && dot) {
                                fdot(ddot, T->n, &T->sh[0][0], m.atm, m.natm, m.bas, m.nbas, m.env, gam, ng, bc);
                        }
                        double t3 = now();
                        tq = t1 - t0 < tq ? t1 - t0 : tq;
                        tb = t2 - t1 < tb ? t2 - t1 : tb;
                        td = t3 - t2 < td ? t3 - t2 : td;
                }
                if (ok && dot) {
                        for (long n = 0; n < (long)T->n * ng * ncomp; n++) {
                                double v = rdot[n], d = fabs(v - ddot[n]);
                                ddmax = fmax(ddmax, d);
                                dvmax = fmax(dvmax, fabs(v));
                        }
                        td_all += td * (double)T->count / T->n;
                } else if (dot) {
                        td_all += tq * (double)T->count / T->n;
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
        if (dot) {
                printf("_dot against int2e_%s_sph contracted here: max|d| %.2e (max|v| %.2e)\n", name, ddmax, dvmax);
                if (nsample > 0)
                        printf("with the contractions: int2e_sph %.1f s, batched %.1f s, _dot %.1f s (%.2fx the batched; "
                               "projected one pass, 1 thread)\n", tq_all, tb_all, td_all, tb_all / td_all);
        }
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
