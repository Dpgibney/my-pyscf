/*
 * The J engine's nuclear gradient (my-pyscf workspace, part 24; bench/gpu/exp_prim_screen.py --j-grad copies this file
 * into gpu4pyscf/lib/gvhf-md). For a primitive shell pair P = (i, j) of the J engine's pair list, the charge
 * distribution G_i G_j = sum_tuv E^{ij}_tuv Lambda_tuv(P) moves with the centers A of i and B of j as
 *     d/dA_x (G_i G_j) = 2 a G_{i+1_x} G_j - i_x G_{i-1_x} G_j,   d/dB_x (G_i G_j) = 2 b G_i G_{j+1_x} - j_x G_i G_{j-1_x},
 * whose Hermite expansions reach one order more than the pair's own. Rt_ip1_dot_dm contracts them, with the density
 * D_ij, with the pair's Hermite potential V^P_tuv of that higher order (MD_build_j_ip1, in a layout pair_loc1 of
 * nf3(lij+1) components per pair) and adds the products to the gradients of the two atoms:
 *     g_A,x += sum_ij D_ij sum_tuv [2 a E^{i+1_x,j}_t - i_x E^{i-1_x,j}_t] E^{ij}_u E^{ij}_v V^P_tuv
 * with the pair's prefactor cc = Kab ci cj (halved when i and j are one shell) in E_z, as dm_to_Rt has it.
 */
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"

// the J engine's largest angular momentum (LMAX in j_engine.py): i and j up to LMAX+1, t up to 2*LMAX+1
#define MD_GRAD_LMAX    4
#define MD_GRAD_NI      (MD_GRAD_LMAX+2)
#define MD_GRAD_NT      (2*MD_GRAD_LMAX+2)
#define E_at(E, i, j, t) E[((i)*MD_GRAD_NI+(j))*MD_GRAD_NT+(t)]

__global__ static
void Rt_ip1_dot_dm_kernel(double *grad, double *V, double *dm, RysIntEnvVars envs,
                          uint32_t *bas_ij_idx, int *pair_loc1, int npairs, int *ao_loc)
{
    int pair_ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_ij >= npairs) {
        return;
    }
    double *env = envs.env;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    int li = bas[ish*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh*BAS_SLOTS+ANG_OF];
    int ia = bas[ish*BAS_SLOTS+ATOM_OF];
    int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
    int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
    double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
    double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
    double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
    double aij = ai + aj;
    double aj_aij = aj / aij;
    double xjxi = env[rj+0] - env[ri+0];
    double yjyi = env[rj+1] - env[ri+1];
    double zjzi = env[rj+2] - env[ri+2];
    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
    double xpa = xjxi * aj_aij;
    double ypa = yjyi * aj_aij;
    double zpa = zjzi * aj_aij;
    double xpb = xpa - xjxi;
    double ypb = ypa - yjyi;
    double zpb = zpa - zjzi;
    double theta_ij = ai * aj / aij;
    double Kab = exp(-theta_ij * rr_ij);
    double cc = Kab * ci * cj;
    if (ish == jsh) {
        cc *= .5;
    }
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    size_t Nao = ao_loc[nbas];
    dm += i0 * Nao + j0;
    V += pair_loc1[pair_ij];

    int li1 = li + 1;
    int lj1 = lj + 1;
    int lij1 = li + lj + 1;
    double Ex[MD_GRAD_NI*MD_GRAD_NI*MD_GRAD_NT];
    double Ey[MD_GRAD_NI*MD_GRAD_NI*MD_GRAD_NT];
    double Ez[MD_GRAD_NI*MD_GRAD_NI*MD_GRAD_NT];
    E_at(Ex,0,0,0) = 1.;
    E_at(Ey,0,0,0) = 1.;
    E_at(Ez,0,0,0) = cc;
    for (int t = 1; t <= lij1; t++) {
        E_at(Ex,0,0,t) = 0.;
        E_at(Ey,0,0,t) = 0.;
        E_at(Ez,0,0,t) = 0.;
    }
    for (int j = 1; j <= lj1; j++) {
        E_at(Ex,0,j,0) = xpb * E_at(Ex,0,j-1,0) + E_at(Ex,0,j-1,1);
        E_at(Ey,0,j,0) = ypb * E_at(Ey,0,j-1,0) + E_at(Ey,0,j-1,1);
        E_at(Ez,0,j,0) = zpb * E_at(Ez,0,j-1,0) + E_at(Ez,0,j-1,1);
        for (int t = 1; t <= lij1; t++) {
            double fac = j/(2*aij*t);
            E_at(Ex,0,j,t) = fac * E_at(Ex,0,j-1,t-1);
            E_at(Ey,0,j,t) = fac * E_at(Ey,0,j-1,t-1);
            E_at(Ez,0,j,t) = fac * E_at(Ez,0,j-1,t-1);
        }
    }
    for (int i = 1; i <= li1; i++) {
        E_at(Ex,i,0,0) = xpa * E_at(Ex,i-1,0,0) + E_at(Ex,i-1,0,1);
        E_at(Ey,i,0,0) = ypa * E_at(Ey,i-1,0,0) + E_at(Ey,i-1,0,1);
        E_at(Ez,i,0,0) = zpa * E_at(Ez,i-1,0,0) + E_at(Ez,i-1,0,1);
        for (int t = 1; t <= lij1; t++) {
            double fac = i/(2*aij*t);
            E_at(Ex,i,0,t) = fac * E_at(Ex,i-1,0,t-1);
            E_at(Ey,i,0,t) = fac * E_at(Ey,i-1,0,t-1);
            E_at(Ez,i,0,t) = fac * E_at(Ez,i-1,0,t-1);
        }
    }
    for (int i = 1; i <= li1; i++) {
        // (li+1, lj+1) is never used: the shifted pairs raise one center at a time
        for (int j = 1; j <= (i == li1 ? lj : lj1); j++) {
            E_at(Ex,i,j,0) = xpb * E_at(Ex,i,j-1,0) + E_at(Ex,i,j-1,1);
            E_at(Ey,i,j,0) = ypb * E_at(Ey,i,j-1,0) + E_at(Ey,i,j-1,1);
            E_at(Ez,i,j,0) = zpb * E_at(Ez,i,j-1,0) + E_at(Ez,i,j-1,1);
            for (int t = 1; t <= lij1; t++) {
                double fac = i/(2*aij*t);
                double fac1 = j/(2*aij*t);
                E_at(Ex,i,j,t) = fac*E_at(Ex,i-1,j,t-1) + fac1*E_at(Ex,i,j-1,t-1);
                E_at(Ey,i,j,t) = fac*E_at(Ey,i-1,j,t-1) + fac1*E_at(Ey,i,j-1,t-1);
                E_at(Ez,i,j,t) = fac*E_at(Ez,i-1,j,t-1) + fac1*E_at(Ez,i,j-1,t-1);
            }
        }
    }

    double ga[3] = {0., 0., 0.};
    double gb[3] = {0., 0., 0.};
    double ai2 = 2 * ai;
    double aj2 = 2 * aj;
    for (int ix = li, i = 0; ix >= 0; ix--) {
    for (int iy = li-ix; iy >= 0; iy--, i++) {
        int iz = li - ix - iy;
        for (int jx = lj, j = 0; jx >= 0; jx--) {
        for (int jy = lj-jx; jy >= 0; jy--, j++) {
            int jz = lj - jx - jy;
            double sax = 0., say = 0., saz = 0., sbx = 0., sby = 0., sbz = 0.;
            int n = 0;
            for (int t = 0; t <= lij1; t++) {
                double ex = E_at(Ex,ix,jx,t);
                double eax = ai2 * E_at(Ex,ix+1,jx,t);
                double ebx = aj2 * E_at(Ex,ix,jx+1,t);
                if (ix > 0) eax -= ix * E_at(Ex,ix-1,jx,t);
                if (jx > 0) ebx -= jx * E_at(Ex,ix,jx-1,t);
                for (int u = 0; u <= lij1-t; u++) {
                    double ey = E_at(Ey,iy,jy,u);
                    double eay = ai2 * E_at(Ey,iy+1,jy,u);
                    double eby = aj2 * E_at(Ey,iy,jy+1,u);
                    if (iy > 0) eay -= iy * E_at(Ey,iy-1,jy,u);
                    if (jy > 0) eby -= jy * E_at(Ey,iy,jy-1,u);
                    for (int v = 0; v <= lij1-t-u; v++, n++) {
                        double ez = E_at(Ez,iz,jz,v);
                        double eaz = ai2 * E_at(Ez,iz+1,jz,v);
                        double ebz = aj2 * E_at(Ez,iz,jz+1,v);
                        if (iz > 0) eaz -= iz * E_at(Ez,iz-1,jz,v);
                        if (jz > 0) ebz -= jz * E_at(Ez,iz,jz-1,v);
                        double vt = V[n];
                        double eyz = ey * ez * vt;
                        double exz = ex * ez * vt;
                        double exy = ex * ey * vt;
                        sax += eax * eyz;
                        say += eay * exz;
                        saz += eaz * exy;
                        sbx += ebx * eyz;
                        sby += eby * exz;
                        sbz += ebz * exy;
                    }
                }
            }
            double d = dm[i*Nao+j];
            ga[0] += d * sax;
            ga[1] += d * say;
            ga[2] += d * saz;
            gb[0] += d * sbx;
            gb[1] += d * sby;
            gb[2] += d * sbz;
        } }
    } }
    atomicAdd(grad+ia*3+0, ga[0]);
    atomicAdd(grad+ia*3+1, ga[1]);
    atomicAdd(grad+ia*3+2, ga[2]);
    atomicAdd(grad+ja*3+0, gb[0]);
    atomicAdd(grad+ja*3+1, gb[1]);
    atomicAdd(grad+ja*3+2, gb[2]);
}

extern "C" {
int Rt_ip1_dot_dm(double *grad, double *V, double *dm, RysIntEnvVars *envs,
                  uint32_t *bas_ij_idx, int *pair_loc1, int npairs, int *ao_loc)
{
    int threads = 128;
    int blocks = (npairs + threads - 1) / threads;
    Rt_ip1_dot_dm_kernel<<<blocks, threads>>>(grad, V, dm, *envs, bas_ij_idx, pair_loc1, npairs, ao_loc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in Rt_ip1_dot_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
