// VV10's grid-response gradient term (gdft/vv10.cu, vv10_grad_eval_E_grid_response_offdiagonal_kernel, and the
// post-processing in grad/rks.py get_nlc_exc_full_response) against part 28's symmetric form (node1). The driver builds
// E_Bgr[B][i] = 1.5 sum_{j on atom B} rho_w_j Phi_ij f_ij (r_i - r_j) for B != atom(i), sets E_Bgr[atom(i)][i] to minus
// the sum of the others, and takes de[B] = sum_i rho_w_i E_Bgr[B][i]. With K_ij = rho_w_i rho_w_j Phi_ij f_ij symmetric,
// that is de[B] = 3 sum_{j on B} T_j with T_j = sum_{i on another atom} K_ij (r_i - r_j): each unordered pair adds w = K_ij
// (r_i - r_j) to T_j and -w to T_i, the rotation scheme of part 26's Fock kernel. Checks de against the original on
// synthetic data (points grouped by atom) and against a long double reference, and times both.
// Build: nvcc -O3 -arch=sm_75 -o vv10_grid_check vv10_grid_check.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define VV10_TILE       128
#define VV10_WARPS      (VV10_TILE / 32)

// the original, verbatim
__global__
static void vv10_grad_eval_E_grid_response_offdiagonal_kernel(double* __restrict__ Egr,
                                                              const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                              const double* __restrict__ omega, const double* __restrict__ kappa,
                                                              const int* __restrict__ grid_associated_atom, const int* __restrict__ grid_offsets_of_atom,
                                                              const int natoms, const int i_grid_begin, const int ngrids)
{
    const int i_unoffset = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y;
    if (i_unoffset >= ngrids || B_atom >= natoms)
        return;
    const int i = i_unoffset + i_grid_begin;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 0;
        return;
    }
    if (i_associated_atom == B_atom) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 0;
        return;
    }

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double3 Egr_i = { 0, 0, 0 };

    const int j_B_atom_start = grid_offsets_of_atom[B_atom];
    const int j_B_atom_end = grid_offsets_of_atom[B_atom + 1];

    for (int j = j_B_atom_start; j < j_B_atom_end; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double3 r_ij = { r_i.x - r_j.x, r_i.y - r_j.y, r_i.z - r_j.z };
        const double r_ij2 = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double E_ij = rho_weight_j * Phi_ij;
        const double dPhi_drj_over_Phi = (omega_i * g_ji * g_sum + omega_j * g_ij * g_sum + (omega_i + omega_j) * g_ij * g_ji) * g_ij_ji_sum_1;
        const double Egr_ij = E_ij * dPhi_drj_over_Phi;

        Egr_i.x += Egr_ij * r_ij.x;
        Egr_i.y += Egr_ij * r_ij.y;
        Egr_i.z += Egr_ij * r_ij.z;
    }

    Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 1.5 * Egr_i.x;
    Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 1.5 * Egr_i.y;
    Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 1.5 * Egr_i.z;
}

// the driver's post-processing of a batch, on the GPU: E_Bgr[atom(i)][i] = -sum_B E_Bgr[B][i]; de[B] += sum_i rho_w_i E_Bgr
__global__ static void original_finish(const double *Egr, const double *rho_weight, const int *atom, int natm, int g0,
                                       int n, double *de)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int a = atom[g0 + i];
    double rw = rho_weight[g0 + i];
    for (int x = 0; x < 3; x++) {
        double s = 0;
        for (int B = 0; B < natm; B++) {
            if (B != a) {
                double v = Egr[B * 3 * n + x * n + i];
                s += v;
                atomicAdd(de + B * 3 + x, v * rw);
            }
        }
        atomicAdd(de + a * 3 + x, -s * rw);
    }
}

// part 28: T_j = sum over i on other atoms of K_ij (r_i - r_j), each unordered pair once (zeroed before)
__global__
static void vv10_grad_grid_response_T_kernel(double* __restrict__ T, const double* __restrict__ grid_coord,
                                             const double* __restrict__ rho_weight, const double* __restrict__ omega,
                                             const double* __restrict__ kappa, const int* __restrict__ atom,
                                             const int ngrids, const int jtiles)
{
    const int ntiles = (ngrids + VV10_TILE - 1) / VV10_TILE;
    const int I = blockIdx.x;
    const int J0 = I + blockIdx.y * jtiles;
    if (J0 >= ntiles) {
        return;
    }
    const int J1 = min(J0 + jtiles, ntiles);
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int i = I * VV10_TILE + t;
    double omega_i = 1., kappa_i = 1., rho_weight_i = 0.;
    double3 r_i = {0., 0., 0.};
    int atom_i = -1;
    if (i < ngrids) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        rho_weight_i = rho_weight[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
        atom_i = atom[i];
    }
    double Tix = 0, Tiy = 0, Tiz = 0;

    __shared__ double s_x[VV10_TILE], s_y[VV10_TILE], s_z[VV10_TILE];
    __shared__ double s_o[VV10_TILE], s_k[VV10_TILE], s_w[VV10_TILE];
    __shared__ int s_a[VV10_TILE];
    __shared__ double s_acc[VV10_WARPS][3][VV10_TILE];
    const int next = (lane + 1) % 32;

    for (int J = J0; J < J1; J++) {
        const int jt = J * VV10_TILE + t;
        if (jt < ngrids) {
            s_x[t] = grid_coord[jt * 3 + 0];
            s_y[t] = grid_coord[jt * 3 + 1];
            s_z[t] = grid_coord[jt * 3 + 2];
            s_o[t] = omega[jt];
            s_k[t] = kappa[jt];
            s_w[t] = rho_weight[jt];
            s_a[t] = atom[jt];
        } else {
            s_x[t] = 0.; s_y[t] = 0.; s_z[t] = 0.;
            s_o[t] = 1.; s_k[t] = 1.; s_w[t] = 0.;
            s_a[t] = -2;
        }
        __syncthreads();
        const bool diag = J == I;
        for (int c = 0; c < VV10_WARPS; c++) {
            double Tjx = 0, Tjy = 0, Tjz = 0;
            if (!diag || c >= warp) {
                for (int s = 0; s < 32; s++) {
                    const int jj = c * 32 + (lane + s) % 32;
                    const double dx = r_i.x - s_x[jj];
                    const double dy = r_i.y - s_y[jj];
                    const double dz = r_i.z - s_z[jj];
                    const double omega_j = s_o[jj];
                    const double kappa_j = s_k[jj];
                    const double r_ij2 = dx * dx + dy * dy + dz * dz;
                    const double g_ij = omega_i * r_ij2 + kappa_i;
                    const double g_ji = omega_j * r_ij2 + kappa_j;
                    const double g_sum = g_ij + g_ji;
                    const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
                    const double Phi_ij = -g_ij_ji_sum_1;
                    const double dPhi_drj_over_Phi = (omega_i * g_ji * g_sum + omega_j * g_ij * g_sum + (omega_i + omega_j) * g_ij * g_ji) * g_ij_ji_sum_1;
                    // pairs of points on different atoms; in a diagonal tile j > i (j == i is on one atom anyway)
                    if (s_a[jj] != atom_i && (!diag || jj > t)) {
                        const double K = rho_weight_i * s_w[jj] * Phi_ij * dPhi_drj_over_Phi;
                        Tix -= K * dx;
                        Tiy -= K * dy;
                        Tiz -= K * dz;
                        Tjx += K * dx;
                        Tjy += K * dy;
                        Tjz += K * dz;
                    }
                    Tjx = __shfl_sync(0xffffffff, Tjx, next);
                    Tjy = __shfl_sync(0xffffffff, Tjy, next);
                    Tjz = __shfl_sync(0xffffffff, Tjz, next);
                }
            }
            s_acc[warp][0][c * 32 + lane] = Tjx;
            s_acc[warp][1][c * 32 + lane] = Tjy;
            s_acc[warp][2][c * 32 + lane] = Tjz;
        }
        __syncthreads();
        if (jt < ngrids) {
            double tx = 0, ty = 0, tz = 0;
            for (int k = 0; k < VV10_WARPS; k++) {
                tx += s_acc[k][0][t];
                ty += s_acc[k][1][t];
                tz += s_acc[k][2][t];
            }
            atomicAdd(T + 0 * ngrids + jt, tx);
            atomicAdd(T + 1 * ngrids + jt, ty);
            atomicAdd(T + 2 * ngrids + jt, tz);
        }
        __syncthreads();
    }
    if (i < ngrids) {
        atomicAdd(T + 0 * ngrids + i, Tix);
        atomicAdd(T + 1 * ngrids + i, Tiy);
        atomicAdd(T + 2 * ngrids + i, Tiz);
    }
}

__global__ static void T_to_atoms(const double *T, const int *atom, int n, double *de)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int a = atom[i];
    for (int x = 0; x < 3; x++) atomicAdd(de + a * 3 + x, 3. * T[x * n + i]);
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 60000;
    int natm = argc > 2 ? atoi(argv[2]) : 24;
    // synthetic points grouped by atom: atom a's points around a center, as a sorted grid has them
    double *h = (double *)malloc(n * 6 * sizeof(double));
    int *ha = (int *)malloc(n * sizeof(int));
    int *hoff = (int *)malloc((natm + 1) * sizeof(int));
    srand(4321);
    double ctr[256][3];
    for (int a = 0; a < natm; a++) for (int x = 0; x < 3; x++) ctr[a][x] = 10. * rand() / RAND_MAX - 5.;
    for (int a = 0; a <= natm; a++) hoff[a] = (int)((long)n * a / natm);
    for (int a = 0; a < natm; a++)
        for (int k = hoff[a]; k < hoff[a + 1]; k++) {
            ha[k] = a;
            double r = 4. * pow((double)rand() / RAND_MAX, 2.);
            double u = 2. * rand() / RAND_MAX - 1., ph = 6.283185307179586 * rand() / RAND_MAX, sq = sqrt(1 - u * u);
            h[k * 3 + 0] = ctr[a][0] + r * sq * cos(ph);
            h[k * 3 + 1] = ctr[a][1] + r * sq * sin(ph);
            h[k * 3 + 2] = ctr[a][2] + r * u;
            h[3 * n + k] = 1e-6 * pow(1e4, (double)rand() / RAND_MAX);     // rho * weight
            h[4 * n + k] = 0.05 * pow(100., (double)rand() / RAND_MAX);    // omega
            h[5 * n + k] = 0.1 * pow(20., (double)rand() / RAND_MAX);      // kappa
        }
    double *d, *de0, *de1, *T, *Egr;
    int *da, *doff;
    cudaMalloc(&d, n * 6 * sizeof(double));
    cudaMemcpy(d, h, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&da, n * sizeof(int));
    cudaMemcpy(da, ha, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&doff, (natm + 1) * sizeof(int));
    cudaMemcpy(doff, hoff, (natm + 1) * sizeof(int), cudaMemcpyHostToDevice);
    double *coord = d, *rw = d + 3 * n, *om = d + 4 * n, *ka = d + 5 * n;
    cudaMalloc(&de0, natm * 3 * 8); cudaMalloc(&de1, natm * 3 * 8); cudaMalloc(&T, 3 * n * 8);
    int batch = n;   // the driver's batch: as much as memory allows, here all points
    cudaMalloc(&Egr, (size_t)natm * 3 * batch * 8);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    float t0, t1;
    for (int rep = 0; rep < 2; rep++) {
        cudaMemset(de0, 0, natm * 3 * 8);
        cudaEventRecord(e0);
        for (int g0 = 0; g0 < n; g0 += batch) {
            int g1 = g0 + batch < n ? g0 + batch : n;
            dim3 threads(128);
            dim3 blocks((g1 - g0 + 127) / 128, natm);
            vv10_grad_eval_E_grid_response_offdiagonal_kernel<<<blocks, threads>>>(Egr, coord, rw, om, ka, da, doff, natm, g0, g1 - g0);
            original_finish<<<(g1 - g0 + 127) / 128, 128>>>(Egr, rw, da, natm, g0, g1 - g0, de0);
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t0, e0, e1);
    }
    for (int rep = 0; rep < 2; rep++) {
        cudaMemset(de1, 0, natm * 3 * 8);
        cudaEventRecord(e0);
        cudaMemset(T, 0, 3 * n * 8);
        int ntiles = (n + VV10_TILE - 1) / VV10_TILE, jtiles = 4;
        dim3 blocks(ntiles, (ntiles + jtiles - 1) / jtiles);
        vv10_grad_grid_response_T_kernel<<<blocks, VV10_TILE>>>(T, coord, rw, om, ka, da, n, jtiles);
        T_to_atoms<<<(n + 255) / 256, 256>>>(T, da, n, de1);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t1, e0, e1);
    }
    double *h0 = (double *)malloc(natm * 3 * 8), *h1 = (double *)malloc(natm * 3 * 8);
    cudaMemcpy(h0, de0, natm * 3 * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(h1, de1, natm * 3 * 8, cudaMemcpyDeviceToHost);
    // long double reference of de[B] = 3 sum_{j on B} sum_{i on other atoms} K_ij (r_i - r_j) for the first 3 atoms
    double maxde = 0, d01 = 0, e0r = 0, e1r = 0;
    for (int k = 0; k < natm * 3; k++) { maxde = fmax(maxde, fabs(h0[k])); d01 = fmax(d01, fabs(h1[k] - h0[k])); }
    for (int B = 0; B < 3; B++) {
        long double acc[3] = {0, 0, 0};
        for (int j = hoff[B]; j < hoff[B + 1]; j++)
            for (int i = 0; i < n; i++) {
                if (ha[i] == B) continue;
                long double dx = (long double)h[i*3] - h[j*3], dy = (long double)h[i*3+1] - h[j*3+1], dz = (long double)h[i*3+2] - h[j*3+2];
                long double r2 = dx*dx + dy*dy + dz*dz;
                long double oi = h[4*n+i], oj = h[4*n+j], gij = oi * r2 + h[5*n+i], gji = oj * r2 + h[5*n+j], gs = gij + gji;
                long double inv = 1.0L / (gij * gji * gs);
                long double f = (oi * gji * gs + oj * gij * gs + (oi + oj) * gij * gji) * inv;
                long double K = (long double)h[3*n+i] * h[3*n+j] * -inv * f;
                acc[0] += K * dx; acc[1] += K * dy; acc[2] += K * dz;
            }
        for (int x = 0; x < 3; x++) {
            long double r = 3.0L * acc[x];
            e0r = fmax(e0r, (double)fabsl((h0[B*3+x] - r) / r));
            e1r = fmax(e1r, (double)fabsl((h1[B*3+x] - r) / r));
        }
    }
    printf("n = %d, %d atoms: original (kernel + finish) %.1f ms, symmetric %.1f ms (%.2fx); max |de| %.3e, "
           "max |de_sym - de_orig| %.2e; against long double (3 atoms): original %.1e, symmetric %.1e (relative)\n",
           n, natm, t0, t1, t0 / t1, maxde, d01, e0r, e1r);
    cudaError_t err = cudaGetLastError();
    printf("%s\n", err == cudaSuccess ? "ok" : cudaGetErrorString(err));
    return 0;
}
