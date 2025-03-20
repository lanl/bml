#ifdef BML_USE_MAGMA
#include <stdbool.h>
#include "magma_v2.h"
#endif

#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_submatrix.h"
#include "../bml_types.h"
#include "../dense/bml_allocate_dense.h"
#include "bml_allocate_ellpack.h"
#include "bml_submatrix_ellpack.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Determine element indices for submatrix, given a set of nodes/orbitals.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Hamiltonian matrix A
 * \param B Graph matrix B
 * \param nodelist List of node/orbital indices
 * \param nsize Size of nodelist
 * \param core_halo_index List of core+halo indices
 * \param vsize Size of core_halo_index and number of cores
 * \param double_jump_flag Flag to use double jump (0=no, 1=yes)
 */
void TYPED_FUNC(
    bml_matrix2submatrix_index_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    int *nodelist,
    int nsize,
    int *core_halo_index,
    int *vsize,
    int double_jump_flag)
{
    int l, ll, ii, ls, k;
    int A_N = A->N;
    int A_M = A->M;
    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int B_N = B->N;
    int B_M = B->M;
    int *B_nnz = B->nnz;
    int *B_index = B->index;

    int ix[A_N];

    memset(ix, 0, A_N * sizeof(int));

    l = 0;
    ll = 0;

#ifdef USE_OMP_OFFLOAD
#pragma omp target update from(A_nnz[:A_N], A_index[:A_N*A_M])
#pragma omp target update from(B_nnz[:B_N], B_index[:B_N*B_M])
#endif

    // Cores are first followed by halos
    for (int j = 0; j < nsize; j++)
    {
        ii = nodelist[j];
        if (ix[ii] == 0)
        {
            ix[ii] = ii + 1;
            core_halo_index[l] = ii;
            l++;
            ll++;
        }

    }

    // Collect halo indices from graph
    for (int j = 0; j < nsize; j++)
    {
        ii = nodelist[j];

        for (int jp = 0; jp < B_nnz[ii]; jp++)
        {
            k = B_index[ROWMAJOR(ii, jp, B_N, B_M)];
            if (ix[k] == 0)
            {
                ix[k] = ii + 1;
                core_halo_index[l] = k;
                l++;
            }
        }
    }

    // Add more halo elements from H
    for (int j = 0; j < nsize; j++)
    {
        ii = nodelist[j];

        for (int jp = 0; jp < A_nnz[ii]; jp++)
        {
            k = A_index[ROWMAJOR(ii, jp, A_N, A_M)];
            if (ix[k] == 0)
            {
                ix[k] = ii + 1;
                core_halo_index[l] = k;
                l++;
            }
        }
    }

    // Perform a "double jump" for extra halo elements
    // based on graph, like performing a symbolic X^2
    if (double_jump_flag == 1)
    {
        ls = l;
        for (int j = 0; j < ls; j++)
        {
            ii = core_halo_index[j];

            for (int jp = 0; jp < B_nnz[ii]; jp++)
            {
                k = B_index[ROWMAJOR(ii, jp, B_N, B_M)];
                if (ix[k] == 0)
                {
                    ix[k] = ii + 1;
                    core_halo_index[l] = k;
                    l++;
                }
            }
        }
    }

    vsize[0] = l;
    vsize[1] = ll;
}

/** Determine element indices for submatrix, given a set of nodes/orbitals.
 *
 * \ingroup submatrix_group_C
 *
 * \param B Graph matrix B
 * \param nodelist List of node/orbital indices
 * \param nsize Size of nodelist
 * \param core_halo_index List of core+halo indices
 * \param vsize Size of core_halo_index and number of cores
 * \param double_jump_flag Flag to use double jump (0=no, 1=yes)
 */
void TYPED_FUNC(
    bml_matrix2submatrix_index_graph_ellpack) (
    bml_matrix_ellpack_t * B,
    int *nodelist,
    int nsize,
    int *core_halo_index,
    int *vsize,
    int double_jump_flag)
{
    int l, ll, ii, ls, k;
    int B_N = B->N;
    int B_M = B->M;
    int *B_index = B->index;
    int *B_nnz = B->nnz;
    int ix[B_N];

    memset(ix, 0, B_N * sizeof(int));

    l = 0;
    ll = 0;

#ifdef USE_OMP_OFFLOAD
#pragma omp target update from(B_nnz[:B_N], B_index[:B_N*B_M])
#endif

    // Cores are first followed by halos
    for (int j = 0; j < nsize; j++)
    {
        ii = nodelist[j];
        if (ix[ii] == 0)
        {
            ix[ii] = ii + 1;
            core_halo_index[l] = ii;
            l++;
            ll++;
        }
    }

    // Collext halo indices from graph
    for (int j = 0; j < nsize; j++)
    {
        ii = nodelist[j];

        for (int jp = 0; jp < B_nnz[ii]; jp++)
        {
            k = B_index[ROWMAJOR(ii, jp, B_N, B_M)];
            if (ix[k] == 0)
            {
                ix[k] = ii + 1;
                core_halo_index[l] = k;
                l++;
            }
        }
    }

    // Use graph for double jumps
    if (double_jump_flag == 1)
    {
        ls = l;
        for (int j = 0; j < ls; j++)
        {
            ii = core_halo_index[j];

            for (int jp = 0; jp < B_nnz[ii]; jp++)
            {
                k = B_index[ROWMAJOR(ii, jp, B_N, B_M)];
                if (ix[k] == 0)
                {
                    ix[k] = ii + 1;
                    core_halo_index[l] = k;
                    l++;
                }
            }
        }
    }

    vsize[0] = l;
    vsize[1] = ll;
}

/** Extract a submatrix from a matrix given a set of core+halo rows.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A
 * \param B Submatrix B
 * \param core_halo_index Set of row indices for submatrix
 * \param lsize Number of indices
 */
void TYPED_FUNC(
    bml_matrix2submatrix_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_dense_t * B,
    int *core_halo_index,
    int lsize)
{
    REAL_T *rvalue;

    int B_N = B->N;
#ifdef BML_USE_MAGMA
    REAL_T *B_matrix = bml_allocate_memory(sizeof(REAL_T) * B->N * B->N);
#else
    REAL_T *B_matrix = B->matrix;
#endif

#ifdef USE_OMP_OFFLOAD
    int A_N = A->N;
    int A_M = A->M;
    int *A_nnz = A->nnz;
    int *A_index = A->index;

#pragma omp target update from(A_nnz[:A_N], A_index[:A_N*A_M])
#endif

#pragma omp parallel for     \
    private(rvalue)          \
    shared(core_halo_index)  \
    shared(A, B_matrix, B_N)
    for (int jb = 0; jb < lsize; jb++)
    {
        rvalue = TYPED_FUNC(bml_getVector_ellpack) (A, core_halo_index,
                                                    core_halo_index[jb],
                                                    lsize);
        for (int j = 0; j < lsize; j++)
        {
            B_matrix[ROWMAJOR(jb, j, B_N, B_N)] = rvalue[j];
        }

        bml_free_memory(rvalue);
    }
#ifdef BML_USE_MAGMA
    MAGMA(setmatrix) (B_N, B_N, (MAGMA_T *) B_matrix, B_N,
                      B->matrix, B->ld, bml_queue());
    bml_free_memory(B_matrix);
#endif
#ifdef MKL_GPU
#pragma omp target update to(B_matrix[0:B_N*B_N])
#endif

}

/** Assemble submatrix into a full matrix based on core+halo indices.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Submatrix A
 * \param B Matrix B
 * \param core_halo_index Set of submatrix row indices
 * \param lsize Number of indices
 * \param llsize Number of core positions
 */
void TYPED_FUNC(
    bml_submatrix2matrix_ellpack) (
    bml_matrix_dense_t * A,
    bml_matrix_ellpack_t * B,
    int *core_halo_index,
    int lsize,
    int llsize,
    double threshold)
{
    int A_N = A->N;
#ifdef BML_USE_MAGMA
    REAL_T *A_matrix = bml_allocate_memory(sizeof(REAL_T) * A->N * A->N);
    MAGMA(getmatrix) (A->N, A->N,
                      A->matrix, A->ld, (MAGMA_T *) A_matrix, A->N,
                      bml_queue());
#else
    REAL_T *A_matrix = A->matrix;
#ifdef MKL_GPU
#pragma omp target update from(A_matrix[0:A_N*A_N])
#endif

#endif

    int B_N = B->N;
    int B_M = B->M;
    int *B_nnz = B->nnz;
    int *B_index = B->index;
    REAL_T *B_value = B->value;

    int ii, icol;

#ifdef USE_OMP_OFFLOAD
#pragma omp target update from(B_nnz[:B_N], B_index[:B_N*B_M], B_value[:B_N*B_M])
#endif

#pragma omp parallel for                      \
    private(ii, icol)                         \
    shared(core_halo_index)                   \
    shared(A_N, A_matrix)                     \
    shared(B_N, B_M, B_nnz, B_index, B_value)
    for (int ja = 0; ja < llsize; ja++)
    {
        ii = core_halo_index[ja];

        icol = 0;
        for (int jb = 0; jb < lsize; jb++)
        {
            if (ABS(A_matrix[ROWMAJOR(ja, jb, A_N, A_N)]) > threshold)
            {
                B_index[ROWMAJOR(ii, icol, B_N, B_M)] = core_halo_index[jb];
                B_value[ROWMAJOR(ii, icol, B_N, B_M)] =
                    A_matrix[ROWMAJOR(ja, jb, A_N, A_N)];
                icol++;
            }
        }
        if (icol > B_M)
        {
            LOG_ERROR("Number of non-zeroes per row >= M, Increase M\n");
        }

        B_nnz[ii] = icol;
    }
#ifdef BML_USE_MAGMA
    bml_free_memory(A_matrix);
#endif
#ifdef USE_OMP_OFFLOAD
#pragma omp target update to(B_nnz[:B_N], B_index[:B_N*B_M], B_value[:B_N*B_M])
#endif
}

// Get matching vector of values
void *TYPED_FUNC(
    bml_getVector_ellpack) (
    bml_matrix_ellpack_t * A,
    int *jj,
    int irow,
    int colCnt)
{
    REAL_T ZERO = 0.0;

    int A_N = A->N;
    int A_M = A->M;
    int *A_nnz = A->nnz;
    int *A_index = A->index;
    REAL_T *A_value = A->value;
    REAL_T *rvalue = bml_noinit_allocate_memory(colCnt * sizeof(REAL_T));

    for (int i = 0; i < colCnt; i++)
    {
        for (int j = 0; j < A_nnz[irow]; j++)
        {
            if (A_index[ROWMAJOR(irow, j, A_N, A_M)] == jj[i])
            {
                rvalue[i] = A_value[ROWMAJOR(irow, j, A_N, A_M)];
                break;
            }
            rvalue[i] = ZERO;
        }
    }
    return rvalue;
}

/** Assemble matrix based on groups of rows from a matrix.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A
 * \param hindex Indices of nodes
 * \param ngroups Number of groups
 * \param threshold Threshold for graph
 */
bml_matrix_ellpack_t
    * TYPED_FUNC(bml_group_matrix_ellpack) (bml_matrix_ellpack_t * A,
                                            int *hindex, int ngroups,
                                            double threshold)
{
    int A_N = A->N;
    int A_M = A->M;
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    REAL_T *A_value = A->value;

#ifdef USE_OMP_OFFLOAD
#pragma omp target update from(A_nnz[:A_N], A_index[:A_N*A_M], A_value[:A_N*A_M])
#endif

#if !(defined(__IBMC_) || defined(__ibmxl__))
    int ix[ngroups];

    memset(ix, 0, sizeof(int) * ngroups);
#endif

    int hnode[A_N];
    int hend;

    bml_matrix_dimension_t matrix_dimension = { ngroups, ngroups, ngroups };
    bml_matrix_ellpack_t *B =
        TYPED_FUNC(bml_noinit_matrix_ellpack) (matrix_dimension,
                                               A->distribution_mode);

    int B_N = B->N;
    int B_M = B->M;
    int *B_index = B->index;
    int *B_nnz = B->nnz;
    REAL_T *B_value = B->value;

#pragma omp parallel for               \
    private(hend)                      \
    shared(hindex, hnode, A_N)
    for (int i = 0; i < ngroups; i++)
    {
        if (i == ngroups - 1)
            hend = A_N;
        else
            hend = hindex[i + 1] - 1;
        for (int j = hindex[i] - 1; j < hend; j++)
        {
            hnode[j] = i;
        }
    }

#if defined(__IBMC_) || defined(__ibmxl__)
#pragma omp parallel for                       \
    private(hend)                              \
    shared(hindex, hnode)                      \
    shared(A_nnz, A_index, A_value, A_N, A_M)  \
    shared(B_nnz, B_index, B_value, B_N, B_M)
#else
#pragma omp parallel for                       \
    private(hend)                              \
    shared(hindex, hnode)                      \
    shared(A_nnz, A_index, A_value, A_N, A_M)  \
    shared(B_nnz, B_index, B_value, B_N, B_M)  \
    firstprivate(ix)
#endif

    for (int i = 0; i < B_N; i++)
    {

#if defined(__IBMC_) || defined(__ibmxl__)
        int ix[ngroups];

        memset(ix, 0, sizeof(int) * ngroups);
#endif

        ix[i] = i + 1;
        B_index[ROWMAJOR(i, 0, B_N, B_M)] = i;
        B_value[ROWMAJOR(i, 0, B_N, B_M)] = 1.0;
        B_nnz[i] = 1;
        if (i == B_N - 1)
            hend = A_N;
        else
            hend = hindex[i + 1] - 1;
        for (int j = hindex[i] - 1; j < hend; j++)
        {
            for (int k = 0; k < A_nnz[j]; k++)
            {
                int ii = hnode[A_index[ROWMAJOR(j, k, A_N, A_M)]];
                if (ix[ii] == 0 && ii != i)
                {
                    //printf("row = %d col = %d val = %e\n", j, A_index[ROWMAJOR(j, k, A_N, A_M)], A_value[ROWMAJOR(j, k, A_N, A_M)]);
                    if (is_above_threshold(A_value[ROWMAJOR(j, k, A_N, A_M)],
                                           threshold))
                    {
                        ix[ii] = i + 1;
                        B_index[ROWMAJOR(i, B_nnz[i], B_N, B_M)] = ii;
                        B_value[ROWMAJOR(i, B_nnz[i], B_N, B_M)] = 1.0;
                        B_nnz[i]++;
                    }
                    else
                    {
                        int kk = A_index[ROWMAJOR(j, k, A_N, A_M)];
                        for (int l = 0; l < A_nnz[kk]; l++)
                        {
                            int jj =
                                hnode[A_index[ROWMAJOR(kk, l, A_N, A_M)]];
                            if (jj == i)
                            {
                                //printf("sym row = %d col = %d val = %e\n", kk, A_index[ROWMAJOR(kk, l, A_N, A_M)], A_value[ROWMAJOR(kk, l, A_N, A_M)]);

                                if (is_above_threshold
                                    (A_value[ROWMAJOR(kk, l, A_N, A_M)],
                                     threshold))
                                {
                                    ix[ii] = i + 1;
                                    B_index[ROWMAJOR(i, B_nnz[i], B_N, B_M)] =
                                        ii;
                                    B_value[ROWMAJOR(i, B_nnz[i], B_N, B_M)] =
                                        1.0;
                                    B_nnz[i]++;
                                    break;
                                }
                            }
                        }
                    }
                }

            }
        }
    }

#ifdef USE_OMP_OFFLOAD
#pragma omp target update to(B_nnz[:B_N], B_index[:B_N*B_M], B_value[:B_N*B_M])
#endif

    return B;
}

/** Extract submatrix into new matrix of same format
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A to extract submatrix from
 * \param irow Index of first row to extract
 * \param icol Index of first column to extract
 * \param B_N Number of rows/columns to extract
 * \param B_M Max number of non-zero elemnts/row in exttacted matrix
 */
bml_matrix_ellpack_t
    * TYPED_FUNC(bml_extract_submatrix_ellpack) (bml_matrix_ellpack_t * A,
                                                 int irow, int icol,
                                                 int B_N, int B_M)
{
    int A_N = A->N;
    int A_M = A->M;
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    REAL_T *A_value = A->value;

    bml_matrix_ellpack_t *B;
    B = TYPED_FUNC(bml_zero_matrix_ellpack) (B_N, B_M, A->distribution_mode);

    int *B_index = B->index;
    int *B_nnz = B->nnz;
    REAL_T *B_value = B->value;

    // loop over subset of rows of A
    for (int i = irow; i < irow + B_N; i++)
    {
        for (int jp = 0; jp < A_nnz[i]; jp++)
        {
            int j = A_index[ROWMAJOR(i, jp, A_N, A_M)];
            if (j >= icol && j < icol + B_N)
            {
                int iB = i - irow;
                B_index[ROWMAJOR(i - irow, B_nnz[iB], B_N, B_M)] = j - icol;
                B_value[ROWMAJOR(i - irow, B_nnz[iB], B_N, B_M)] =
                    A_value[ROWMAJOR(i, jp, A_N, A_M)];
                B_nnz[iB]++;
            }
        }
    }

    return B;
}

/** Assign a block B into matrix A
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param irow First row where to insert block B
 * \param icol Offset column to insert block B
 */
void TYPED_FUNC(
    bml_assign_submatrix_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    int irow,
    int icol)
{
    int A_N = A->N;
    int A_M = A->M;
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    REAL_T *A_value = A->value;

    int B_N = B->N;
    int B_M = B->M;
    int *B_index = B->index;
    int *B_nnz = B->nnz;
    REAL_T *B_value = B->value;

    // loop over rows of B
    for (int i = 0; i < B_N; i++)
    {
        for (int jp = 0; jp < B_nnz[i]; jp++)
        {
            int jB = B_index[ROWMAJOR(i, jp, B_N, B_M)];
            int jpA = A_nnz[i + irow];
            A_value[ROWMAJOR(i + irow, jpA, A_N, A_M)] =
                B_value[ROWMAJOR(i, jp, B_N, B_M)];
            A_index[ROWMAJOR(i + irow, jpA, A_N, A_M)] = jB + icol;
            A_nnz[i + irow]++;
        }
    }
}
