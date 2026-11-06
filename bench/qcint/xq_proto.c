/*
 * Prototype: Rys ERIs of 8 shell quartets of one class at once, the quartets in the SIMD lanes.
 * qcint puts the primitive combinations of one quartet in its lanes, so a quartet of single-primitive
 * shells (all d and f shells of def2-TZVP(D)) runs on its scalar path. Here: class (dd|dd), one
 * primitive per shell, spherical, timed and compared against int2e_sph of a qcint build on the same
 * quartets of a dump_basis.py molecule (the s8-unique quartets with q_ij q_kl > 1e-13, as qbench).
 *
 *   xq_proto LIB.so MOL.bin [--reps N] [--lr]      (--lr: int2e with omega = 0.3)
 *
 * The roots come from qcint's own CINTrys_roots (its sources are compiled in, see the Makefile line at
 * the end), point by point; the rest works on 8 quartets at a time with GCC vector types.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int CINTrys_roots(int nroots, double x, double *u, double *w);

typedef double v8 __attribute__((vector_size(64), aligned(64)));
#define LANES 8
#define NROOTS 5
#define NSPH 625
#define BAS(slot, i) bas[(i) * 8 + (slot)]
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define PTR_EXP 5
#define PTR_COEFF 6
#define PTR_COORD 1
#define PTR_RANGE_OMEGA 8

typedef int (*FINT)(double *out, int *dims, int *shls, int *atm, int natm, int *bas, int nbas,
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

// libcint's cartesian d functions xx, xy, xz, yy, yz, zz, and its d cart -> sph matrix
static const int CX[6] = {2, 1, 1, 0, 0, 0};
static const int CY[6] = {0, 1, 0, 2, 1, 0};
static const int CZ[6] = {0, 0, 1, 0, 1, 2};
static const double C2S[5][6] = {
        {0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0925484305920792, 0.0},
        {-0.31539156525252, 0.0, 0.0, -0.31539156525252, 0.0, 0.63078313050504},
        {0.0, 0.0, 1.0925484305920792, 0.0, 0.0, 0.0},
        {0.5462742152960396, 0.0, 0.0, -0.5462742152960396, 0.0, 0.0},
};

typedef struct {
        v8 ai, aj, ak, al, cc;
        v8 ri[3], rj[3], rk[3], rl[3];
} Batch;

static void gather(Batch *b, Mol *m, int (*sh)[4], int n)
{
        int *bas = m->bas, *atm = m->atm;
        double *env = m->env;
        for (int lane = 0; lane < LANES; lane++) {
                int *s = sh[lane < n ? lane : n - 1];  // pad the last batch with its last quartet
                double c = 1;
                for (int t = 0; t < 4; t++) {
                        double a = env[BAS(PTR_EXP, s[t])];
                        c *= env[BAS(PTR_COEFF, s[t])];
                        const double *r = env + atm[BAS(ATOM_OF, s[t]) * 6 + PTR_COORD];
                        v8 *R = t == 0 ? b->ri : t == 1 ? b->rj : t == 2 ? b->rk : b->rl;
                        for (int d = 0; d < 3; d++) R[d][lane] = r[d];
                        if (t == 0) b->ai[lane] = a;
                        else if (t == 1) b->aj[lane] = a;
                        else if (t == 2) b->ak[lane] = a;
                        else b->al[lane] = a;
                }
                b->cc[lane] = c;
        }
}

static v8 I[NROOTS][3][3][3][3][3];  // [root][dir][i][j][k][l]
static v8 CART[6][6][6][6];          // [lc][kc][jc][ic]
static v8 S1[5][6][6][6];            // [ls][kc][jc][ic]
static v8 S2[5][5][6][6];            // [ls][ks][jc][ic]
static v8 S3[5][5][5][6];            // [ls][ks][js][ic]

// (dd|dd) of the 8 quartets of b; out[lane * 625 + i + 5 * (j + 5 * (k + 5 * l))], as int2e_sph
static void eri_dddd(const Batch *b, double omega, double *out)
{
        const double common = M_PI * M_PI * M_PI * 2 / sqrt(M_PI);
        v8 aij = b->ai + b->aj, akl = b->ak + b->al;
        v8 rij[3], rkl[3], pa[3], qc[3], pq[3];
        v8 rrij = {0}, rrkl = {0}, rpq = {0};
        for (int d = 0; d < 3; d++) {
                rij[d] = b->ri[d] - b->rj[d];
                rkl[d] = b->rk[d] - b->rl[d];
                rrij += rij[d] * rij[d];
                rrkl += rkl[d] * rkl[d];
                v8 p = (b->ai * b->ri[d] + b->aj * b->rj[d]) / aij;
                v8 q = (b->ak * b->rk[d] + b->al * b->rl[d]) / akl;
                pa[d] = p - b->ri[d];
                qc[d] = q - b->rk[d];
                pq[d] = p - q;
                rpq += pq[d] * pq[d];
        }
        v8 a1 = aij * akl, a0 = a1 / (aij + akl), fac1;
        double u[NROOTS][LANES] __attribute__((aligned(64)));
        double w[NROOTS][LANES] __attribute__((aligned(64)));
        v8 theta = a0 - a0 + 1;
        if (omega > 0) {  // long range: as libcint, x and t^2 scaled by theta, the weights by sqrt(theta)
                v8 om2 = a0 - a0 + omega * omega;
                theta = om2 / (om2 + a0);
        }
        for (int lane = 0; lane < LANES; lane++) {
                double eij = exp(-b->ai[lane] * b->aj[lane] / aij[lane] * rrij[lane]);
                double ekl = exp(-b->ak[lane] * b->al[lane] / akl[lane] * rrkl[lane]);
                double a1l = a1[lane];
                fac1[lane] = sqrt(a0[lane] / (a1l * a1l * a1l)) * common * b->cc[lane] * eij * ekl
                        * sqrt(theta[lane]);
                double ul[NROOTS], wl[NROOTS];
                CINTrys_roots(NROOTS, a0[lane] * theta[lane] * rpq[lane], ul, wl);
                for (int r = 0; r < NROOTS; r++) {
                        // libcint's u, as t^2/(1-t^2); long range: t^2 -> theta t^2
                        double t2 = ul[r] / (ul[r] + 1) * theta[lane];
                        u[r][lane] = t2 / (1 - t2);
                        w[r][lane] = wl[r];
                }
        }
        for (int r = 0; r < NROOTS; r++) {
                v8 ur, wr;
                memcpy(&ur, u[r], sizeof(v8));
                memcpy(&wr, w[r], sizeof(v8));
                // libcint's CINTg0_2e_2d coefficients
                v8 u2 = a0 * ur;
                v8 tmp4 = .5 / (u2 * (aij + akl) + a1);
                v8 b00 = u2 * tmp4;
                v8 tmp1 = 2 * b00, tmp2 = tmp1 * akl, tmp3 = tmp1 * aij;
                v8 b10 = b00 + tmp4 * akl;
                v8 b01 = b00 + tmp4 * aij;
                for (int d = 0; d < 3; d++) {
                        v8 c00 = pa[d] - tmp2 * pq[d];
                        v8 c0p = qc[d] + tmp3 * pq[d];
                        v8 g[5][5];  // [n][m]: (n 0 | m 0), n <= li + lj, m <= lk + ll
                        g[0][0] = d == 2 ? wr * fac1 : a0 - a0 + 1;
                        g[1][0] = c00 * g[0][0];
                        for (int n = 1; n < 4; n++)
                                g[n + 1][0] = c00 * g[n][0] + n * b10 * g[n - 1][0];
                        for (int m = 0; m < 4; m++) {
                                v8 bm = m * b01;
                                g[0][m + 1] = c0p * g[0][m] + (m ? bm * g[0][m - 1] : 0 * c0p);
                                for (int n = 1; n < 5; n++)
                                        g[n][m + 1] = c0p * g[n][m] + (m ? bm * g[n][m - 1] : 0 * c0p)
                                                + n * b00 * g[n - 1][m];
                        }
                        // bra HRR (i, j+1| = (i+1, j| + R_ij (i, j|
                        v8 h1[4][5], h2[3][5];
                        for (int i = 0; i < 4; i++)
                                for (int m = 0; m < 5; m++)
                                        h1[i][m] = g[i + 1][m] + rij[d] * g[i][m];
                        for (int i = 0; i < 3; i++)
                                for (int m = 0; m < 5; m++)
                                        h2[i][m] = h1[i + 1][m] + rij[d] * h1[i][m];
                        for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++) {
                                const v8 *s = j == 0 ? g[i] : j == 1 ? h1[i] : h2[i];
                                v8 k1[4], k2[3];
                                for (int k = 0; k < 4; k++) k1[k] = s[k + 1] + rkl[d] * s[k];
                                for (int k = 0; k < 3; k++) k2[k] = k1[k + 1] + rkl[d] * k1[k];
                                for (int k = 0; k < 3; k++) {
                                        I[r][d][i][j][k][0] = s[k];
                                        I[r][d][i][j][k][1] = k1[k];
                                        I[r][d][i][j][k][2] = k2[k];
                                }
                        }
                }
        }
        for (int lc = 0; lc < 6; lc++)
        for (int kc = 0; kc < 6; kc++)
        for (int jc = 0; jc < 6; jc++)
        for (int ic = 0; ic < 6; ic++) {
                v8 s = {0};
                for (int r = 0; r < NROOTS; r++)
                        s += I[r][0][CX[ic]][CX[jc]][CX[kc]][CX[lc]]
                           * I[r][1][CY[ic]][CY[jc]][CY[kc]][CY[lc]]
                           * I[r][2][CZ[ic]][CZ[jc]][CZ[kc]][CZ[lc]];
                CART[lc][kc][jc][ic] = s;
        }
        // cart -> sph, one index at a time
        for (int ls = 0; ls < 5; ls++)
        for (int kc = 0; kc < 6; kc++)
        for (int jc = 0; jc < 6; jc++)
        for (int ic = 0; ic < 6; ic++) {
                v8 s = {0};
                for (int lc = 0; lc < 6; lc++) if (C2S[ls][lc] != 0) s += C2S[ls][lc] * CART[lc][kc][jc][ic];
                S1[ls][kc][jc][ic] = s;
        }
        for (int ls = 0; ls < 5; ls++)
        for (int ks = 0; ks < 5; ks++)
        for (int jc = 0; jc < 6; jc++)
        for (int ic = 0; ic < 6; ic++) {
                v8 s = {0};
                for (int kc = 0; kc < 6; kc++) if (C2S[ks][kc] != 0) s += C2S[ks][kc] * S1[ls][kc][jc][ic];
                S2[ls][ks][jc][ic] = s;
        }
        for (int ls = 0; ls < 5; ls++)
        for (int ks = 0; ks < 5; ks++)
        for (int js = 0; js < 5; js++)
        for (int ic = 0; ic < 6; ic++) {
                v8 s = {0};
                for (int jc = 0; jc < 6; jc++) if (C2S[js][jc] != 0) s += C2S[js][jc] * S2[ls][ks][jc][ic];
                S3[ls][ks][js][ic] = s;
        }
        for (int ls = 0; ls < 5; ls++)
        for (int ks = 0; ks < 5; ks++)
        for (int js = 0; js < 5; js++)
        for (int is = 0; is < 5; is++) {
                v8 s = {0};
                for (int ic = 0; ic < 6; ic++) if (C2S[is][ic] != 0) s += C2S[is][ic] * S3[ls][ks][js][ic];
                int o = is + 5 * (js + 5 * (ks + 5 * ls));
                for (int lane = 0; lane < LANES; lane++) out[lane * NSPH + o] = s[lane];
        }
}

int main(int argc, char **argv)
{
        if (argc < 3) {
                fprintf(stderr, "usage: xq_proto LIB.so MOL.bin [--reps N] [--lr]\n");
                return 1;
        }
        Mol m = load(argv[2]);
        int reps = 3;
        double omega = 0;
        for (int a = 3; a < argc; a++) {
                if (!strcmp(argv[a], "--reps")) reps = atoi(argv[++a]);
                else if (!strcmp(argv[a], "--lr")) omega = 0.3;
        }
        m.env[PTR_RANGE_OMEGA] = omega;
        void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (lib == NULL) {
                fprintf(stderr, "%s\n", dlerror());
                return 1;
        }
        FINT fint = (FINT)dlsym(lib, "int2e_sph");
        FOPT fopt = (FOPT)dlsym(lib, "int2e_optimizer");
        void *opt = NULL;
        fopt(&opt, m.atm, m.natm, m.bas, m.nbas, m.env);

        // the s8-unique (dd|dd) quartets of single-primitive shells with q_ij q_kl > 1e-13
        int *bas = m.bas, nbas = m.nbas;
        int cap = 1 << 20, nq = 0;
        int (*sh)[4] = malloc(sizeof(int[4]) * cap);
        for (int i = 0; i < nbas; i++)
        for (int j = 0; j <= i; j++)
        for (int k = 0; k <= i; k++)
        for (int l = 0; l <= k; l++) {
                if (k == i && l > j) break;
                int s[4] = {i, j, k, l}, ok = 1;
                for (int t = 0; t < 4; t++)
                        ok &= BAS(ANG_OF, s[t]) == 2 && BAS(NPRIM_OF, s[t]) == 1 && BAS(NCTR_OF, s[t]) == 1;
                if (!ok || m.q[i * nbas + j] * m.q[k * nbas + l] <= 1e-13) continue;
                if (nq == cap) sh = realloc(sh, sizeof(int[4]) * (cap *= 2));
                memcpy(sh[nq++], s, sizeof(s));
        }
        printf("(dd|dd), one primitive per shell%s: %d quartets\n", omega > 0 ? ", long range" : "", nq);

        double *ref = malloc(sizeof(double) * NSPH * (size_t)nq);
        double *cache = malloc(sizeof(double) * (1 << 22));
        double *bout = aligned_alloc(64, sizeof(double) * NSPH * LANES);
        Batch b;
        double tq = 1e300, tx = 1e300;
        for (int rep = 0; rep < reps; rep++) {
                double t0 = now();
                for (int n = 0; n < nq; n++)
                        fint(ref + (size_t)NSPH * n, NULL, sh[n], m.atm, m.natm, m.bas, m.nbas, m.env, opt, cache);
                double t1 = now();
                double sink = 0;
                for (int n = 0; n < nq; n += LANES) {
                        int nb = nq - n < LANES ? nq - n : LANES;
                        gather(&b, &m, sh + n, nb);
                        eri_dddd(&b, omega, bout);
                        sink += bout[0];
                }
                double t2 = now();
                if (sink == 12345.6789) printf("!");
                if (t1 - t0 < tq) tq = t1 - t0;
                if (t2 - t1 < tx) tx = t2 - t1;
        }
        // compare every quartet
        double dmax = 0, vmax = 0, rmax = 0;
        for (int n = 0; n < nq; n += LANES) {
                int nb = nq - n < LANES ? nq - n : LANES;
                gather(&b, &m, sh + n, nb);
                eri_dddd(&b, omega, bout);
                for (int lane = 0; lane < nb; lane++) {
                        double qmax = 0, qd = 0;
                        for (int o = 0; o < NSPH; o++) {
                                double v = ref[(size_t)NSPH * (n + lane) + o];
                                qmax = fmax(qmax, fabs(v));
                                qd = fmax(qd, fabs(v - bout[lane * NSPH + o]));
                        }
                        vmax = fmax(vmax, qmax);
                        dmax = fmax(dmax, qd);
                        rmax = fmax(rmax, qd / fmax(qmax, 1e-6));
                }
        }
        printf("qcint int2e_sph: %.3f s, %.0f ns/quartet\n", tq, 1e9 * tq / nq);
        printf("8 quartets/vector: %.3f s, %.0f ns/quartet  (%.2fx)\n", tx, 1e9 * tx / nq, tq / tx);
        printf("max|d| %.2e (max|v| %.2e), max|d|/max|quartet| %.2e\n", dmax, vmax, rmax);
        return 0;
}
// Build (qcint's root code compiled in, with qcint's flags):
// Q=../../qcint/src B=../../build/qcint-pgo
// gcc -DKEEP_GOING -DMIN_EXPCUTOFF=20 -DWITH_F12 -I$B/include -I$B/src -I$Q -march=native -O3 -ffast-math \
//     -fno-math-errno -std=gnu99 -o xq_proto xq_proto.c $Q/rys_roots.c $Q/polyfits.c $Q/rys_wheeler.c \
//     $Q/find_roots.c $Q/eigh.c $Q/fmt.c $Q/misc.c -ldl -lquadmath -lm
