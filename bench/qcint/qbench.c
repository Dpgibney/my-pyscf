/*
 * Time a libcint/qcint build on the unique shell quartets of a molecule, grouped by class
 * (angular momenta and number of primitive combinations), or compare its integrals with another
 * build. The library is loaded with dlopen, so one binary serves every variant.
 *
 *   qbench LIB.so MOL.bin [MODE] [--thr T] [--reps N]   per-class timings, single thread
 *   qbench LIB.so MOL.bin [MODE] --compare LIB2.so      max differences over all quartets
 * MODE: int2e (default), ip1 (gradients), ipip1, ip1ip2, ipvip1 (Hessians), cart (int2e_cart),
 *       sr, lr (int2e with omega = -0.3 / +0.3)
 *
 * MOL.bin is written by dump_basis.py. Quartets are those with q_ij q_kl > T (default 1e-13, the
 * SCF Schwarz bound): s8-unique for int2e, all ij with kl s2 for ip1 (gradients).
 * --compare reports the largest difference and the largest relative to the largest value of its quartet
 * (or 1e-6).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
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
#define NKEY (8 * 8 * 8 * 8 * 3)
#define CACHE_DOUBLES (1 << 22)

typedef struct {
        int natm, nbas, nenv;
        int *atm, *bas;
        double *env, *q;
} Mol;

typedef struct {
        int n, cap;
        int *sh;        // 4 per quartet
        double best;    // seconds
        long nint;      // integrals (functions) per pass
        long nprim;     // primitive combinations per pass
} Class;

static void *xmalloc(size_t n)
{
        void *p = malloc(n);
        if (p == NULL) {
                fprintf(stderr, "malloc(%zu) failed\n", n);
                exit(1);
        }
        return p;
}

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
        m.atm = xmalloc(sizeof(int) * m.natm * 6);
        m.bas = xmalloc(sizeof(int) * m.nbas * 8);
        m.env = xmalloc(sizeof(double) * m.nenv);
        m.q = xmalloc(sizeof(double) * m.nbas * m.nbas);
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

static void *sym(void *lib, const char *name)
{
        void *p = dlsym(lib, name);
        if (p == NULL) {
                fprintf(stderr, "%s\n", dlerror());
                exit(1);
        }
        return p;
}

static int CART = 0;
static int nfunc(int *bas, int sh, int ip1)
{
        (void)ip1;
        int l = BAS(ANG_OF, sh);
        return (CART ? (l + 1) * (l + 2) / 2 : l * 2 + 1) * BAS(NCTR_OF, sh);
}

static int key_of(int *bas, int *sh)
{
        long np = 1;
        int k = 0, n;
        for (n = 0; n < 4; n++) {
                k = k * 8 + BAS(ANG_OF, sh[n]);
                np *= BAS(NPRIM_OF, sh[n]);
        }
        // primitive combinations: 1 (scalar path), 2-7 (partly filled SIMD), >= 8
        return k * 3 + (np == 1 ? 0 : np < 8 ? 1 : 2);
}

static void add(Class *c, int *bas, int *sh, int ncomp)
{
        Class *p = c + key_of(bas, sh);
        if (p->n == p->cap) {
                p->cap = p->cap ? p->cap * 2 : 64;
                p->sh = realloc(p->sh, sizeof(int) * 4 * p->cap);
        }
        memcpy(p->sh + 4 * p->n, sh, sizeof(int) * 4);
        p->n++;
        long ni = (long)ncomp, np = 1;
        int n;
        for (n = 0; n < 4; n++) {
                ni *= nfunc(bas, sh[n], 0);
                np *= BAS(NPRIM_OF, sh[n]);
        }
        p->nint += ni;
        p->nprim += np;
}

static Class *quartets(Mol *m, int ip1, double thr, int ncomp)
{
        Class *c = calloc(NKEY, sizeof(Class));
        int nbas = m->nbas, *bas = m->bas;
        double *q = m->q;
        int sh[4], i, j, k, l;
        for (i = 0; i < nbas; i++)
        for (j = 0; j < (ip1 ? nbas : i + 1); j++)
        for (k = 0; k < (ip1 ? nbas : i + 1); k++)
        for (l = 0; l <= k; l++) {
                if (!ip1 && k == i && l > j) break;
                if (q[i * nbas + j] * q[k * nbas + l] <= thr) continue;
                sh[0] = i; sh[1] = j; sh[2] = k; sh[3] = l;
                add(c, bas, sh, ncomp);
        }
        return c;
}

static double now(void)
{
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return t.tv_sec + 1e-9 * t.tv_nsec;
}

static const char *LNAME = "spdfghik";

int main(int argc, char **argv)
{
        if (argc < 3) {
                fprintf(stderr, "usage: qbench LIB.so MOL.bin [int2e|ip1] [--thr T] [--reps N] [--compare LIB2.so]\n");
                return 1;
        }
        const char *libpath = argv[1];
        Mol m = load(argv[2]);
        int ip1 = 0, reps = 3, a;
        double thr = 1e-13;
        const char *cmp = NULL;
        const char *fname = "int2e_sph", *oname = "int2e_optimizer";
        int ncomp = 1;
        for (a = 3; a < argc; a++) {
                if (!strcmp(argv[a], "ip1")) {
                        ip1 = 1; ncomp = 3; fname = "int2e_ip1_sph"; oname = "int2e_ip1_optimizer";
                } else if (!strcmp(argv[a], "ipip1")) {
                        ip1 = 1; ncomp = 9; fname = "int2e_ipip1_sph"; oname = "int2e_ipip1_optimizer";
                } else if (!strcmp(argv[a], "ip1ip2")) {
                        ip1 = 1; ncomp = 9; fname = "int2e_ip1ip2_sph"; oname = "int2e_ip1ip2_optimizer";
                } else if (!strcmp(argv[a], "ipvip1")) {
                        ip1 = 1; ncomp = 9; fname = "int2e_ipvip1_sph"; oname = "int2e_ipvip1_optimizer";
                } else if (!strcmp(argv[a], "cart")) {
                        CART = 1; fname = "int2e_cart";
                } else if (!strcmp(argv[a], "sr")) {
                        m.env[8] = -0.3;  // PTR_RANGE_OMEGA
                } else if (!strcmp(argv[a], "lr")) {
                        m.env[8] = 0.3;
                } else if (!strcmp(argv[a], "int2e")) {
                } else if (!strcmp(argv[a], "--thr")) thr = atof(argv[++a]);
                else if (!strcmp(argv[a], "--reps")) reps = atoi(argv[++a]);
                else if (!strcmp(argv[a], "--compare")) cmp = argv[++a];
        }

        void *lib = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
        if (lib == NULL) {
                fprintf(stderr, "%s\n", dlerror());
                return 1;
        }
        FINT fint = (FINT)sym(lib, fname);
        FOPT fopt = (FOPT)sym(lib, oname);
        void *opt = NULL;
        fopt(&opt, m.atm, m.natm, m.bas, m.nbas, m.env);

        Class *c = quartets(&m, ip1, thr, ncomp);
        double *buf = xmalloc(sizeof(double) * (1 << 20));
        double *cache = xmalloc(sizeof(double) * CACHE_DOUBLES);
        int key, n;
        long nq = 0, nint = 0;
        for (key = 0; key < NKEY; key++) {
                nq += c[key].n;
                nint += c[key].nint;
        }

        if (cmp != NULL) {
                // the second build in its own namespace: both export the same symbols and soname
                void *lib2 = dlmopen(LM_ID_NEWLM, cmp, RTLD_NOW | RTLD_LOCAL);
                if (lib2 == NULL) {
                        fprintf(stderr, "%s\n", dlerror());
                        return 1;
                }
                FINT fint2 = (FINT)sym(lib2, fname);
                FOPT fopt2 = (FOPT)sym(lib2, oname);
                void *opt2 = NULL;
                fopt2(&opt2, m.atm, m.natm, m.bas, m.nbas, m.env);
                double *buf2 = xmalloc(sizeof(double) * (1 << 20));
                double dmax = 0, vmax = 0, rmax = 0, rq = 0, rd = 0;
                long nz1 = 0, nz2 = 0;
                int worst[4] = {0}, rworst[4] = {0};
                for (key = 0; key < NKEY; key++) {
                        for (n = 0; n < c[key].n; n++) {
                                int *sh = c[key].sh + 4 * n;
                                long ni = ncomp, i;
                                int s;
                                for (s = 0; s < 4; s++) ni *= nfunc(m.bas, sh[s], ip1);
                                nz1 += fint(buf, NULL, sh, m.atm, m.natm, m.bas, m.nbas, m.env, opt, cache);
                                nz2 += fint2(buf2, NULL, sh, m.atm, m.natm, m.bas, m.nbas, m.env, opt2, cache);
                                double qmax = 0, qd = 0;
                                for (i = 0; i < ni; i++) {
                                        qmax = fmax(qmax, fabs(buf[i]));
                                        qd = fmax(qd, fabs(buf[i] - buf2[i]));
                                }
                                vmax = fmax(vmax, qmax);
                                if (qd > dmax) {
                                        dmax = qd;
                                        memcpy(worst, sh, sizeof(worst));
                                }
                                // relative to the largest integral of the quartet, or 1e-6 for
                                // quartets that small (their values can be pure rounding)
                                if (qd / fmax(qmax, 1e-6) > rmax) {
                                        rmax = qd / fmax(qmax, 1e-6);
                                        rq = qmax;
                                        rd = qd;
                                        memcpy(rworst, sh, sizeof(rworst));
                                }
                        }
                }
                printf("%s %ld quartets, %ld integrals: max|d| %.2e (max|v| %.2e) at (%d %d|%d %d), "
                       "max|d|/max|quartet| %.2e (%.1e of %.1e at (%d %d|%d %d)), nonzero flags %ld vs %ld\n",
                       fname, nq, nint, dmax, vmax, worst[0], worst[1], worst[2], worst[3], rmax,
                       rd, rq, rworst[0], rworst[1], rworst[2], rworst[3], nz1, nz2);
                return 0;
        }

        double total_best = 0;
        int r;
        for (key = 0; key < NKEY; key++) c[key].best = 1e300;
        for (r = 0; r < reps; r++) {
                for (key = 0; key < NKEY; key++) {
                        if (c[key].n == 0) continue;
                        double t0 = now();
                        for (n = 0; n < c[key].n; n++) {
                                fint(buf, NULL, c[key].sh + 4 * n, m.atm, m.natm, m.bas, m.nbas, m.env,
                                     opt, cache);
                        }
                        double t = now() - t0;
                        if (t < c[key].best) c[key].best = t;
                }
        }
        for (key = 0; key < NKEY; key++) {
                if (c[key].n) total_best += c[key].best;
        }
        // classes sorted by time
        int order[NKEY], nk = 0, x, y;
        for (key = 0; key < NKEY; key++) {
                if (c[key].n) order[nk++] = key;
        }
        for (x = 0; x < nk; x++)
        for (y = x + 1; y < nk; y++) {
                if (c[order[y]].best > c[order[x]].best) {
                        int t = order[x]; order[x] = order[y]; order[y] = t;
                }
        }
        double tb[3] = {0, 0, 0};
        long nb[3] = {0, 0, 0};
        for (x = 0; x < nk; x++) {
                tb[order[x] % 3] += c[order[x]].best;
                nb[order[x] % 3] += c[order[x]].n;
        }
        printf("%s: %ld quartets, %ld integrals, %.3f s per pass (best of %d, 1 thread), %.1f ns/integral\n",
               fname, nq, nint, total_best, reps, 1e9 * total_best / nint);
        printf("by primitive combinations: 1: %.1f%% of time (%ld quartets), 2-7: %.1f%% (%ld), >=8: %.1f%% (%ld)\n",
               100 * tb[0] / total_best, nb[0], 100 * tb[1] / total_best, nb[1],
               100 * tb[2] / total_best, nb[2]);
        printf("%-8s %-5s %9s %8s %9s %9s %10s\n", "class", "prims", "quartets", "time%", "ns/quart",
               "ns/integ", "prims/qrt");
        double cum = 0;
        for (x = 0; x < nk && cum < .97 * total_best; x++) {
                key = order[x];
                Class *p = c + key;
                int k4 = key / 3, b = key % 3;
                printf("(%c%c|%c%c)  %-5s %9d %7.1f%% %9.0f %9.2f %10.1f\n",
                       LNAME[k4 / 512], LNAME[k4 / 64 % 8], LNAME[k4 / 8 % 8], LNAME[k4 % 8],
                       b == 0 ? "1" : b == 1 ? "2-7" : ">=8", p->n, 100 * p->best / total_best,
                       1e9 * p->best / p->n, 1e9 * p->best / p->nint, (double)p->nprim / p->n);
                cum += p->best;
        }
        return 0;
}
