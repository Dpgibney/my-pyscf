/*
 * Prototype, generic: Rys ERIs of 8 shell quartets of one type at once, the quartets in the SIMD lanes
 * (qcint puts the primitive combinations of one quartet there). Any l <= 3, segmented contractions
 * (summed per lane over the primitive combinations), spherical, full range or long range.
 *
 * Projects one pass over all s8-unique quartets with q_ij q_kl > 1e-13 of a dump_basis.py molecule:
 * quartets are grouped by type (l and number of primitives of each shell, in quartet order), every
 * type is sampled evenly (up to --sample quartets), qcint's int2e_sph and the prototype are timed on
 * the same samples (best of --reps), and each type's time is scaled by its count. Every sampled
 * integral is compared.
 *
 *   xq_generic LIB.so MOL.bin [--lr] [--sample N] [--reps N]
 *
 * Build: the comment at the end (qcint's root code compiled in, point by point roots).
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
#define MAXL 3
#define MAXNC 10
#define MAXNS 7
#define MAXR 7
#define MAXN 7
#define MAXP 16
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

// libcint's cart -> sph matrices as PySCF's gto.cart2sph(l) gives them (with the s and p factors)
static const double C2S0[1][1] = {{0.28209479177387814}};
static const double C2S1[3][3] = {{0.4886025119029199, 0, 0}, {0, 0.4886025119029199, 0},
                                  {0, 0, 0.4886025119029199}};
static const double C2S2[5][6] = {
        {0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0925484305920792, 0.0},
        {-0.31539156525252, 0.0, 0.0, -0.31539156525252, 0.0, 0.63078313050504},
        {0.0, 0.0, 1.0925484305920792, 0.0, 0.0, 0.0},
        {0.5462742152960396, 0.0, 0.0, -0.5462742152960396, 0.0, 0.0}};
static const double C2S3[7][10] = {
        {0.0, 1.7701307697799304, 0.0, 0.0, 0.0, 0.0, -0.5900435899266435, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 2.8906114426405543, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0},
        {0.0, 0.0, -1.1195289977703462, 0.0, 0.0, 0.0, 0.0, -1.1195289977703462, 0.0, 0.7463526651802308},
        {-0.4570457994644657, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.4453057213202771, 0.0, 0.0, 0.0, 0.0, -1.4453057213202771, 0.0, 0.0},
        {0.5900435899266435, 0.0, 0.0, -1.7701307697799304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
static double C2S[MAXL + 1][MAXNS][MAXNC];
static const int CART[MAXL + 1][MAXNC][3] = {  // libcint's order of the cartesian functions
        {{0, 0, 0}},
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}},
        {{3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2}, {0, 3, 0}, {0, 2, 1}, {0, 1, 2},
         {0, 0, 3}}};

static int NC(int l) { return (l + 1) * (l + 2) / 2; }
static int NS(int l) { return 2 * l + 1; }

static void init_tables(void)
{
        for (int l = 0; l <= MAXL; l++) {
                for (int s = 0; s < NS(l); s++)
                for (int c = 0; c < NC(l); c++)
                        C2S[l][s][c] = l == 0 ? C2S0[s][c] : l == 1 ? C2S1[s][c] : l == 2 ? C2S2[s][c] : C2S3[s][c];
        }
}

// the shells of one batch: per lane, exponents and coefficients of each primitive, centres
typedef struct {
        int l[4], np[4];
        v8 a[4][MAXP], c[4][MAXP], r[4][3];
} Batch;

static void gather(Batch *b, Mol *m, int (*sh)[4], int n)
{
        int *bas = m->bas, *atm = m->atm;
        double *env = m->env;
        for (int t = 0; t < 4; t++) {
                b->l[t] = BAS(ANG_OF, sh[0][t]);
                b->np[t] = BAS(NPRIM_OF, sh[0][t]);
        }
        for (int lane = 0; lane < LANES; lane++) {
                int *s = sh[lane < n ? lane : n - 1];  // the last batch is padded with its last quartet
                for (int t = 0; t < 4; t++) {
                        const double *e = env + BAS(PTR_EXP, s[t]), *c = env + BAS(PTR_COEFF, s[t]);
                        for (int p = 0; p < b->np[t]; p++) {
                                b->a[t][p][lane] = e[p];
                                b->c[t][p][lane] = c[p];
                        }
                        const double *r = env + atm[BAS(ATOM_OF, s[t]) * 6 + PTR_COORD];
                        for (int d = 0; d < 3; d++) b->r[t][d][lane] = r[d];
                }
        }
}

static v8 I[MAXR][3][4][4][4][4];          // [root][dir][i][j][k][l]
static v8 GC[MAXNC * MAXNC * MAXNC * MAXNC];  // cartesian, [lc][kc][jc][ic]
static v8 T1[MAXNC * MAXNC * MAXNC * MAXNC], T2[MAXNC * MAXNC * MAXNC * MAXNC];

static inline __attribute__((always_inline))
void eri_t(const Batch *b, double omega, double *out, const int li, const int lj, const int lk, const int ll)
{
        const double common = M_PI * M_PI * M_PI * 2 / sqrt(M_PI);
        const int nroots = (li + lj + lk + ll) / 2 + 1, nmax = li + lj, mmax = lk + ll;
        const int nci = NC(li), ncj = NC(lj), nck = NC(lk), ncl = NC(ll);
        const int ncart = nci * ncj * nck * ncl;
        // offsets of each cartesian function's power in I's [i][j][k][l] (4 x 4 x 4 x 4)
        int ox[4][MAXNC], oy[4][MAXNC], oz[4][MAXNC];
        const int stride[4] = {64, 16, 4, 1};
        for (int t = 0; t < 4; t++)
                for (int c = 0; c < NC(b->l[t]); c++) {
                        ox[t][c] = CART[b->l[t]][c][0] * stride[t];
                        oy[t][c] = CART[b->l[t]][c][1] * stride[t];
                        oz[t][c] = CART[b->l[t]][c][2] * stride[t];
                }
        v8 rij[3], rkl[3], rrij = {0}, rrkl = {0};
        for (int d = 0; d < 3; d++) {
                rij[d] = b->r[0][d] - b->r[1][d];
                rkl[d] = b->r[2][d] - b->r[3][d];
                rrij += rij[d] * rij[d];
                rrkl += rkl[d] * rkl[d];
        }
        for (int n = 0; n < ncart; n++) GC[n] = (v8){0};
        for (int pi = 0; pi < b->np[0]; pi++)
        for (int pj = 0; pj < b->np[1]; pj++)
        for (int pk = 0; pk < b->np[2]; pk++)
        for (int pl = 0; pl < b->np[3]; pl++) {
                v8 ai = b->a[0][pi], aj = b->a[1][pj], ak = b->a[2][pk], al = b->a[3][pl];
                v8 cc = b->c[0][pi] * b->c[1][pj] * b->c[2][pk] * b->c[3][pl];
                v8 aij = ai + aj, akl = ak + al;
                v8 pa[3], qc[3], pq[3], rpq = {0};
                for (int d = 0; d < 3; d++) {
                        v8 p = (ai * b->r[0][d] + aj * b->r[1][d]) / aij;
                        v8 q = (ak * b->r[2][d] + al * b->r[3][d]) / akl;
                        pa[d] = p - b->r[0][d];
                        qc[d] = q - b->r[2][d];
                        pq[d] = p - q;
                        rpq += pq[d] * pq[d];
                }
                v8 a1 = aij * akl, a0 = a1 / (aij + akl), fac1;
                double u[MAXR][LANES] __attribute__((aligned(64)));
                double w[MAXR][LANES] __attribute__((aligned(64)));
                for (int lane = 0; lane < LANES; lane++) {
                        double theta = 1;
                        if (omega > 0) theta = omega * omega / (omega * omega + a0[lane]);
                        double eij = exp(-ai[lane] * aj[lane] / aij[lane] * rrij[lane]);
                        double ekl = exp(-ak[lane] * al[lane] / akl[lane] * rrkl[lane]);
                        double a1l = a1[lane];
                        fac1[lane] = sqrt(a0[lane] / (a1l * a1l * a1l) * theta) * common * cc[lane] * eij * ekl;
                        double ul[MAXR], wl[MAXR];
                        CINTrys_roots(nroots, a0[lane] * theta * rpq[lane], ul, wl);
                        for (int r = 0; r < nroots; r++) {
                                double t2 = ul[r] / (ul[r] + 1) * theta;
                                u[r][lane] = t2 / (1 - t2);
                                w[r][lane] = wl[r];
                        }
                }
                for (int r = 0; r < nroots; r++) {
                        v8 ur, wr;
                        memcpy(&ur, u[r], sizeof(v8));
                        memcpy(&wr, w[r], sizeof(v8));
                        v8 u2 = a0 * ur;
                        v8 tmp4 = .5 / (u2 * (aij + akl) + a1);
                        v8 b00 = u2 * tmp4;
                        v8 tmp1 = 2 * b00, tmp2 = tmp1 * akl, tmp3 = tmp1 * aij;
                        v8 b10 = b00 + tmp4 * akl;
                        v8 b01 = b00 + tmp4 * aij;
                        for (int d = 0; d < 3; d++) {
                                v8 c00 = pa[d] - tmp2 * pq[d];
                                v8 c0p = qc[d] + tmp3 * pq[d];
                                v8 g[MAXN][MAXN];  // [n][m]
                                g[0][0] = d == 2 ? wr * fac1 : a0 - a0 + 1;
                                if (nmax > 0) g[1][0] = c00 * g[0][0];
                                for (int n = 1; n < nmax; n++)
                                        g[n + 1][0] = c00 * g[n][0] + n * b10 * g[n - 1][0];
                                for (int m = 0; m < mmax; m++)
                                        for (int n = 0; n <= nmax; n++) {
                                                v8 v = c0p * g[n][m];
                                                if (m) v += m * b01 * g[n][m - 1];
                                                if (n) v += n * b00 * g[n - 1][m];
                                                g[n][m + 1] = v;
                                        }
                                v8 h[MAXL + 1][MAXN][MAXN];  // [j][i][m], bra HRR
                                for (int n = 0; n <= nmax; n++)
                                        for (int m = 0; m <= mmax; m++) h[0][n][m] = g[n][m];
                                for (int j = 1; j <= lj; j++)
                                        for (int i = 0; i <= nmax - j; i++)
                                                for (int m = 0; m <= mmax; m++)
                                                        h[j][i][m] = h[j - 1][i + 1][m] + rij[d] * h[j - 1][i][m];
                                for (int i = 0; i <= li; i++)
                                for (int j = 0; j <= lj; j++) {
                                        v8 kk[MAXL + 1][MAXN];  // [l][k], ket HRR
                                        for (int m = 0; m <= mmax; m++) kk[0][m] = h[j][i][m];
                                        for (int l = 1; l <= ll; l++)
                                                for (int k = 0; k <= mmax - l; k++)
                                                        kk[l][k] = kk[l - 1][k + 1] + rkl[d] * kk[l - 1][k];
                                        for (int k = 0; k <= lk; k++)
                                                for (int l = 0; l <= ll; l++) I[r][d][i][j][k][l] = kk[l][k];
                                }
                        }
                }
                v8 *gc = GC;
                for (int lc = 0; lc < ncl; lc++)
                for (int kc = 0; kc < nck; kc++)
                for (int jc = 0; jc < ncj; jc++)
                for (int ic = 0; ic < nci; ic++) {
                        int x = ox[0][ic] + ox[1][jc] + ox[2][kc] + ox[3][lc];
                        int y = oy[0][ic] + oy[1][jc] + oy[2][kc] + oy[3][lc];
                        int z = oz[0][ic] + oz[1][jc] + oz[2][kc] + oz[3][lc];
                        v8 s = *gc;
                        for (int r = 0; r < nroots; r++) {
                                const v8 *Ir = &I[r][0][0][0][0][0];
                                s += Ir[x] * Ir[256 + y] * Ir[512 + z];
                        }
                        *gc++ = s;
                }
        }
        // cart -> sph, l, k, j, i in turn; A[lc][kc][jc][ic] -> ... -> [ls][ks][js][is]
        const int nsi = NS(li), nsj = NS(lj), nsk = NS(lk), nsl = NS(ll);
        for (int ls = 0; ls < nsl; ls++)
        for (int x = 0; x < nck * ncj * nci; x++) {
                v8 s = {0};
                for (int lc = 0; lc < ncl; lc++)
                        if (C2S[ll][ls][lc] != 0) s += C2S[ll][ls][lc] * GC[lc * nck * ncj * nci + x];
                T1[ls * nck * ncj * nci + x] = s;
        }
        for (int ls = 0; ls < nsl; ls++)
        for (int ks = 0; ks < nsk; ks++)
        for (int x = 0; x < ncj * nci; x++) {
                v8 s = {0};
                for (int kc = 0; kc < nck; kc++)
                        if (C2S[lk][ks][kc] != 0) s += C2S[lk][ks][kc] * T1[(ls * nck + kc) * ncj * nci + x];
                T2[(ls * nsk + ks) * ncj * nci + x] = s;
        }
        for (int lks = 0; lks < nsl * nsk; lks++)
        for (int js = 0; js < nsj; js++)
        for (int ic = 0; ic < nci; ic++) {
                v8 s = {0};
                for (int jc = 0; jc < ncj; jc++)
                        if (C2S[lj][js][jc] != 0) s += C2S[lj][js][jc] * T2[(lks * ncj + jc) * nci + ic];
                T1[(lks * nsj + js) * nci + ic] = s;
        }
        const int nsph = nsi * nsj * nsk * nsl;
        for (int ljks = 0; ljks < nsl * nsk * nsj; ljks++)
        for (int is = 0; is < nsi; is++) {
                v8 s = {0};
                for (int ic = 0; ic < nci; ic++)
                        if (C2S[li][is][ic] != 0) s += C2S[li][is][ic] * T1[ljks * nci + ic];
                for (int lane = 0; lane < LANES; lane++) out[lane * nsph + ljks * nsi + is] = s[lane];
        }
}

// one specialized copy of eri_t per class: compile-time l, so its loops have constant bounds
#define XQ_CASE(a, b_, c, d) case (a) << 6 | (b_) << 4 | (c) << 2 | (d): eri_t(b, omega, out, a, b_, c, d); break;
#define XQ_L(a, b_, c) XQ_CASE(a, b_, c, 0) XQ_CASE(a, b_, c, 1) XQ_CASE(a, b_, c, 2) XQ_CASE(a, b_, c, 3)
#define XQ_K(a, b_) XQ_L(a, b_, 0) XQ_L(a, b_, 1) XQ_L(a, b_, 2) XQ_L(a, b_, 3)
#define XQ_J(a) XQ_K(a, 0) XQ_K(a, 1) XQ_K(a, 2) XQ_K(a, 3)
static void eri(const Batch *b, double omega, double *out)
{
        switch (b->l[0] << 6 | b->l[1] << 4 | b->l[2] << 2 | b->l[3]) {
        XQ_J(0) XQ_J(1) XQ_J(2) XQ_J(3)
        }
}

typedef struct {
        long count;
        int n, cap;
        int (*sh)[4];
} Type;

int main(int argc, char **argv)
{
        if (argc < 3) {
                fprintf(stderr, "usage: xq_generic LIB.so MOL.bin [--lr] [--sample N] [--reps N]\n");
                return 1;
        }
        init_tables();
        Mol m = load(argv[2]);
        int reps = 2, nsample = 4000;
        double omega = 0;
        for (int a = 3; a < argc; a++) {
                if (!strcmp(argv[a], "--reps")) reps = atoi(argv[++a]);
                else if (!strcmp(argv[a], "--sample")) nsample = atoi(argv[++a]);
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

        int *bas = m.bas, nbas = m.nbas;
        // shell types (l, nprim); quartet types are 4 shell types in quartet order
        int stype[4096], ntype = 0, tl[64], tp[64];
        for (int s = 0; s < nbas; s++) {
                if (BAS(NCTR_OF, s) != 1 || BAS(ANG_OF, s) > MAXL || BAS(NPRIM_OF, s) > MAXP) {
                        fprintf(stderr, "shell %d: l %d nprim %d nctr %d not handled\n", s,
                                BAS(ANG_OF, s), BAS(NPRIM_OF, s), BAS(NCTR_OF, s));
                        return 1;
                }
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
                for (int j = 0; j <= i; j++)
                for (int k = 0; k <= i; k++)
                for (int l = 0; l <= k; l++) {
                        if (k == i && l > j) break;
                        if (m.q[i * nbas + j] * m.q[k * nbas + l] <= 1e-13) continue;
                        Type *t = qt + ((stype[i] * ntype + stype[j]) * ntype + stype[k]) * ntype + stype[l];
                        if (pass == 0) {
                                t->count++;
                                nq++;
                                continue;
                        }
                        // every stride-th quartet of the type
                        long stride = t->count > nsample ? t->count / nsample : 1;
                        long seen = t->cap++;  // cap counts the quartets seen in pass 1
                        if (seen % stride == 0 && t->n < nsample) {
                                if (t->sh == NULL) t->sh = malloc(sizeof(int[4]) * nsample);
                                int *s = t->sh[t->n++];
                                s[0] = i; s[1] = j; s[2] = k; s[3] = l;
                        }
                }
        }
        printf("%s: %ld quartets, %d shell types; up to %d sampled per quartet type, best of %d\n",
               omega > 0 ? "long range (omega 0.3)" : "full range", nq, ntype, nsample, reps);

        double *ref = malloc(sizeof(double) * 2401 * (size_t)nsample);
        double *cache = malloc(sizeof(double) * (1 << 22));
        double *bout = aligned_alloc(64, sizeof(double) * 2401 * LANES);
        Batch *b = aligned_alloc(64, sizeof(Batch));
        double tq_all = 0, tx_all = 0, dmax = 0, vmax = 0;
        double cat_q[3][3] = {{0}}, cat_x[3][3] = {{0}};  // [max l: <=1, 2, 3][primitive combinations: 1, 2-7, >=8]
        for (int t = 0; t < nqt; t++) {
                Type *T = qt + t;
                if (T->n == 0) continue;
                int *s0 = T->sh[0], lmax = 0;
                long ncomb = 1, nsph = 1;
                for (int x = 0; x < 4; x++) {
                        int l = BAS(ANG_OF, s0[x]);
                        lmax = l > lmax ? l : lmax;
                        ncomb *= BAS(NPRIM_OF, s0[x]);
                        nsph *= NS(l);
                }
                double tq = 1e300, tx = 1e300;
                for (int rep = 0; rep < reps; rep++) {
                        double t0 = now();
                        for (int n = 0; n < T->n; n++)
                                fint(ref + nsph * n, NULL, T->sh[n], m.atm, m.natm, m.bas, m.nbas, m.env, opt, cache);
                        double t1 = now();
                        double sink = 0;
                        for (int n = 0; n < T->n; n += LANES) {
                                int nb = T->n - n < LANES ? T->n - n : LANES;
                                gather(b, &m, T->sh + n, nb);
                                eri(b, omega, bout);
                                sink += bout[0];
                        }
                        double t2 = now();
                        if (sink == 12345.6789) printf("!");
                        tq = t1 - t0 < tq ? t1 - t0 : tq;
                        tx = t2 - t1 < tx ? t2 - t1 : tx;
                }
                for (int n = 0; n < T->n; n += LANES) {
                        int nb = T->n - n < LANES ? T->n - n : LANES;
                        gather(b, &m, T->sh + n, nb);
                        eri(b, omega, bout);
                        for (int lane = 0; lane < nb; lane++)
                                for (int o = 0; o < nsph; o++) {
                                        double v = ref[nsph * (n + lane) + o];
                                        vmax = fmax(vmax, fabs(v));
                                        dmax = fmax(dmax, fabs(v - bout[lane * nsph + o]));
                                }
                }
                double scale = (double)T->count / T->n;
                int cl = lmax <= 1 ? 0 : lmax - 1, cp = ncomb == 1 ? 0 : ncomb < 8 ? 1 : 2;
                cat_q[cl][cp] += tq * scale;
                cat_x[cl][cp] += tx * scale;
                tq_all += tq * scale;
                tx_all += tx * scale;
        }
        const char *lname[3] = {"max l <= 1", "max l = 2", "max l = 3"};
        const char *pname[3] = {"1 prim. comb.", "2-7", ">= 8"};
        printf("%-12s %-14s %12s %12s %7s\n", "", "", "qcint (s)", "8/vector (s)", "ratio");
        for (int cl = 0; cl < 3; cl++)
                for (int cp = 0; cp < 3; cp++)
                        if (cat_q[cl][cp] > 0)
                                printf("%-12s %-14s %12.1f %12.1f %6.2fx\n", lname[cl], pname[cp], cat_q[cl][cp],
                                       cat_x[cl][cp], cat_q[cl][cp] / cat_x[cl][cp]);
        printf("%-27s %12.1f %12.1f %6.2fx   (projected one pass, 1 thread)\n", "all", tq_all, tx_all, tq_all / tx_all);
        printf("sampled integrals: max|d| %.2e (max|v| %.2e)\n", dmax, vmax);
        return 0;
}
// Build (qcint's root code compiled in, with qcint's flags):
// Q=../../qcint/src B=../../build/qcint-pgo
// gcc -DKEEP_GOING -DMIN_EXPCUTOFF=20 -DWITH_F12 -I$B/include -I$B/src -I$Q -march=native -O3 -ffast-math \
//     -fno-math-errno -std=gnu99 -o xq_generic xq_generic.c $Q/rys_roots.c $Q/polyfits.c $Q/rys_wheeler.c \
//     $Q/find_roots.c $Q/eigh.c $Q/fmt.c $Q/misc.c -ldl -lquadmath -lm
