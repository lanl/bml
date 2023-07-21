#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_types.h"
#include "../bml_domain.h"
#include "bml_allocate_ellpack.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef BML_USE_ROCSPARSE
// Copy rocsparse headers into src/C-interface/rocsparse/ and edit rocsparse_functions.h to remove '[[...]]' text
#include "../rocsparse/rocsparse.h"
/* DEBUG
#include <hip/hip_runtime.h> // needed for hipDeviceSynchronize()
*/
#endif

/** Deallocate a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void TYPED_FUNC(
    bml_deallocate_ellpack) (
    bml_matrix_ellpack_t * A)
{
#ifdef USE_OMP_OFFLOAD
    int N = A->N;
    int M = A->M;

    int *A_nnz = A->nnz;
    int *A_index = A->index;
    REAL_T *A_value = A->value;

#pragma omp target exit data map(delete: A_nnz[:N], A_index[:N*M], A_value[:N*M])

#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE)
    int *csrColInd = A->csrColInd;
    int *csrRowPtr = A->csrRowPtr;
    REAL_T *csrVal = A->csrVal;
#pragma omp target exit data map(delete:csrVal[:N*M], csrColInd[:N*M], csrRowPtr[:N+1])
#endif
#endif

    bml_deallocate_domain(A->domain);
    bml_free_memory(A->value);
    bml_free_memory(A->index);
    bml_free_memory(A->nnz);

#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE)
    bml_free_memory(A->csrRowPtr);
    bml_free_memory(A->csrColInd);
    bml_free_memory(A->csrVal);
#endif

    bml_free_memory(A);
}

/** Clear a matrix.
 *
 * Numbers of non-zeroes, indices, and values are set to zero.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void TYPED_FUNC(
    bml_clear_ellpack) (
    bml_matrix_ellpack_t * A)
{
    REAL_T *A_value = A->value;
#if defined (USE_OMP_OFFLOAD)
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    int N = A->N;
    int M = A->M;

#pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++)
    {
        A_nnz[i] = 0;
    }

#pragma omp target teams distribute parallel for collapse(2) schedule (static, 1)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            A_index[ROWMAJOR(i, j, N, M)] = 0;
            A_value[ROWMAJOR(i, j, N, M)] = 0.0;
        }
    }
#else // conditional for offload

#ifdef INTEL_OPT
#pragma omp parallel for simd
#pragma vector aligned
    for (int i = 0; i < (A->N * A->M); i++)
    {
        __assume_aligned(A->index, MALLOC_ALIGNMENT);
        __assume_aligned(A_value, MALLOC_ALIGNMENT);
        A->index[i] = 0;
        A_value[i] = 0.0;
    }

#pragma omp parallel for simd
#pragma vector aligned
    for (int i = 0; i < A->N; i++)
    {
        __assume_aligned(A->nnz, MALLOC_ALIGNMENT);
        A->nnz[i] = 0;
    }
#else
    memset(A->nnz, 0, A->N * sizeof(int));
    memset(A->index, 0, A->N * A->M * sizeof(int));
    memset(A_value, 0.0, A->N * A->M * sizeof(REAL_T));
#endif

#endif // conditional for offload
}

/** Allocate a matrix with uninitialized values.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param matrix_dimension The matrix size.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t
    * TYPED_FUNC(bml_noinit_matrix_ellpack) (bml_matrix_dimension_t
                                             matrix_dimension,
                                             bml_distribution_mode_t
                                             distrib_mode)
{
    bml_matrix_ellpack_t *A =
        bml_noinit_allocate_memory(sizeof(bml_matrix_ellpack_t));
    A->matrix_type = ellpack;
    A->matrix_precision = MATRIX_PRECISION;
    A->N = matrix_dimension.N_rows;
    A->M = matrix_dimension.N_nz_max;
    A->distribution_mode = distrib_mode;
    A->index = bml_noinit_allocate_memory(sizeof(int) * A->N * A->M);
    A->nnz = bml_allocate_memory(sizeof(int) * A->N);
    A->value = bml_noinit_allocate_memory(sizeof(REAL_T) * A->N * A->M);
    A->domain = bml_default_domain(A->N, A->M, distrib_mode);

#if defined(USE_OMP_OFFLOAD)
    int N = A->N;
    int M = A->M;
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    REAL_T *A_value = A->value;

#pragma omp target enter data map(alloc:A_value[:N*M], A_index[:N*M], A_nnz[:N])
#pragma omp target update to(A_value[:N*M], A_index[:N*M], A_nnz[:N])
#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE)
    A->csrColInd = bml_noinit_allocate_memory(sizeof(int) * N * M);
    A->csrRowPtr = bml_allocate_memory(sizeof(int) * (N + 1));
    A->csrVal = bml_noinit_allocate_memory(sizeof(REAL_T) * N * M);

    int *csrColInd = A->csrColInd;
    int *csrRowPtr = A->csrRowPtr;
    REAL_T *csrVal = A->csrVal;
#pragma omp target enter data map(alloc:csrVal[:N*M], csrColInd[:N*M])
#pragma omp target enter data map(to:csrRowPtr[:N+1])
#endif
#endif

    return A;
}

/** Allocate the zero matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_zero_matrix_ellpack) (
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    assert(M > 0);

    bml_matrix_ellpack_t *A =
        bml_allocate_memory(sizeof(bml_matrix_ellpack_t));
    A->matrix_type = ellpack;
    A->matrix_precision = MATRIX_PRECISION;
    A->N = N;
    A->M = M;
    A->distribution_mode = distrib_mode;
    // need to keep these allocates for host copy
    A->index = bml_allocate_memory(sizeof(int) * N * M);
    A->nnz = bml_allocate_memory(sizeof(int) * N);
    A->value = bml_allocate_memory(sizeof(REAL_T) * N * M);
#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE)
    A->csrColInd = bml_allocate_memory(sizeof(int) * N * M);
    A->csrRowPtr = bml_allocate_memory(sizeof(int) * (N + 1));
    A->csrVal = bml_allocate_memory(sizeof(REAL_T) * N * M);
#endif

    A->domain = bml_default_domain(N, M, distrib_mode);

#if defined(USE_OMP_OFFLOAD)
    REAL_T *A_value = A->value;
    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int NM = N * M;
#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE)
    int *csrColInd = A->csrColInd;
    int *csrRowPtr = A->csrRowPtr;
    REAL_T *csrVal = A->csrVal;
#endif
#pragma omp target enter data map(alloc:A_value[:N*M], A_index[:N*M], A_nnz[:N])

#pragma omp target teams distribute parallel for schedule (static, 1)
    for (int i = 0; i < N; i++)
    {
        A_nnz[i] = 0;
    }

#pragma omp target teams distribute parallel for collapse(2) schedule (static, 1)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            A_index[ROWMAJOR(i, j, N, M)] = 0;
            A_value[ROWMAJOR(i, j, N, M)] = 0.0;
        }
    }

#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE)
#pragma omp target enter data map(to:csrVal[:N*M], csrColInd[:N*M], csrRowPtr[:N+1])
#endif
#endif
    return A;
}

/** Allocate a banded random matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_banded_matrix_ellpack) (
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    bml_matrix_ellpack_t *A =
        TYPED_FUNC(bml_zero_matrix_ellpack) (N, M, distrib_mode);

    REAL_T *A_value = A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    const REAL_T INV_RAND_MAX = 1.0 / (REAL_T) RAND_MAX;

#pragma omp parallel for shared(A_value, A_index, A_nnz)
    for (int i = 0; i < N; i++)
    {
        int jind = 0;
        for (int j = (i - M / 2 >= 0 ? i - M / 2 : 0);
             j < (i - M / 2 + M <= N ? i - M / 2 + M : N); j++)
        {
            A_value[ROWMAJOR(i, jind, N, M)] = rand() * INV_RAND_MAX;
            A_index[ROWMAJOR(i, jind, N, M)] = j;
            jind++;
        }
        A_nnz[i] = jind;
    }
#if defined(USE_OMP_OFFLOAD)
#pragma omp target update to(A_value[:N*M], A_index[:N*M], A_nnz[:N])
#endif
    return A;
}

/** Allocate a random matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 * 
 * NOTE: 
 *  The resulting nonzero structure is not necessarily uniform since we 
 *  are sampling between [0, N], N << RAND_MAX. The diagonal entry is 
 *  stored first.
 *  
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 *
 *  Note: Do not use OpenMP when setting values for a random matrix,
 *  this makes the operation non-repeatable.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_random_matrix_ellpack) (
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    bml_matrix_ellpack_t *A =
        TYPED_FUNC(bml_zero_matrix_ellpack) (N, M, distrib_mode);

    int *col_marker = bml_allocate_memory(sizeof(int) * N );
    int *col_marker_pos = bml_allocate_memory(sizeof(int) * M );
    
    REAL_T *A_value = A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    const REAL_T INV_RAND_MAX = 1.0 / (REAL_T) RAND_MAX;

/* initialize col_marker */
    for (int j = 0; j < N; j++)
    {
        col_marker[j] = -1;
    }
    for (int i = 0; i < N; i++)
    {
        int col = i;
        int nnz_row = 0;
        for (int j = 0; j < M; j++)
        {
            if(col_marker[col] == -1)
            {
                A_value[ROWMAJOR(i, nnz_row, N, M)] = rand() * INV_RAND_MAX;
                A_index[ROWMAJOR(i, nnz_row, N, M)] = col;
                /* save position of col_marker */
                col_marker_pos[nnz_row] = col;
                /* mark column index position */
                col_marker[col] = 1;
                nnz_row++;
            }
            col = rand() % (N + 1);
        }
        /* update nnz of row */
        A_nnz[i] = nnz_row;
        /* reset col_marker */
        for (int j = 0; j < nnz_row; j++)
        {
            col_marker[col_marker_pos[j]] = -1; 
        }
    }
    /* free memory */
    bml_free_memory(col_marker);
    bml_free_memory(col_marker_pos);

#if defined(USE_OMP_OFFLOAD)
#pragma omp target update to(A_value[:N*M], A_index[:N*M], A_nnz[:N])
#endif
    return A;
}

/** Allocate the identity matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_identity_matrix_ellpack) (
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    bml_matrix_ellpack_t *A =
        TYPED_FUNC(bml_zero_matrix_ellpack) (N, M, distrib_mode);

    REAL_T *A_value = A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;

#pragma omp parallel for shared(A_value, A_index, A_nnz)
    for (int i = 0; i < N; i++)
    {
#ifdef INTEL_OPT
        __assume_aligned(A_value, MALLOC_ALIGNMENT);
        __assume_aligned(A_index, MALLOC_ALIGNMENT);
        __assume_aligned(A_nnz, MALLOC_ALIGNMENT);
#endif
        A_value[ROWMAJOR(i, 0, N, M)] = (REAL_T) 1.0;
        A_index[ROWMAJOR(i, 0, N, M)] = i;
        A_nnz[i] = 1;
    }
#if defined(USE_OMP_OFFLOAD)
#pragma omp target update to(A_value[:N*M], A_index[:N*M], A_nnz[:N])
#endif
    return A;
}

#if defined(BML_USE_ROCSPARSE)
/** Sort the column indices of a rocsparse CSR matrix
 *
 *
 *  \ingroup
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
void TYPED_FUNC(
    bml_sort_rocsparse_ellpack) (
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A)
{
    int N = A->N;
    int M = A->M;
    int *csrColInd = A->csrColInd;
    int *csrRowPtr = A->csrRowPtr;
    REAL_T *csrVal = A->csrVal;
    int nnz;

    int i;
    
    REAL_T *csrVal_tmp;

    rocsparse_mat_descr mat;
      
    size_t lworkInBytes = 0;
    char *dwork = NULL;

    nnz = csrRowPtr[N];
    // Temporary array for sorting
    
    csrVal_tmp = (REAL_T *) malloc(sizeof(REAL_T)* nnz);

        // rocSPARSE APIs
    rocsparse_status status = rocsparse_status_success;

    rocsparse_datatype computeType = BML_ROCSPARSE_T;

    BML_CHECK_ROCSPARSE(rocsparse_create_mat_descr
			(&mat));

        // Sort the matrix
#pragma omp target data use_device_ptr(csrRowPtr, csrColInd)
    {
      BML_CHECK_ROCSPARSE(rocsparse_csrsort_buffer_size
			  (handle, N, N,
			   nnz, csrRowPtr,
			   csrColInd, &lworkInBytes));
    }
    dwork = (char *) malloc(sizeof(char)*lworkInBytes);
    rocsparse_int *perm;
    perm = (rocsparse_int *) malloc(nnz * sizeof(rocsparse_int));
    REAL_T *csrVal_tmp_sorted;
    csrVal_tmp = (REAL_T *) malloc(nnz * sizeof(REAL_T));
#pragma omp target enter data map(alloc:dwork[:lworkInBytes],perm[:nnz],csrVal_tmp[:nnz])

#pragma omp target teams distribute parallel for
    for (i=0; i<nnz; i++) {
      csrVal_tmp[i] = csrVal[i];
    }
#pragma omp target data use_device_ptr(csrRowPtr,	\
				       csrColInd, perm, dwork,	\
				       csrVal_tmp, csrVal)
    {
      BML_CHECK_ROCSPARSE(rocsparse_create_identity_permutation
			  (handle, nnz, perm));
      BML_CHECK_ROCSPARSE(rocsparse_csrsort
			  (handle, N, N,
			   nnz,
			   (rocsparse_mat_descr) mat,
			   csrRowPtr, csrColInd, perm,
			   dwork));
      BML_CHECK_ROCSPARSE(bml_rocsparse_xgthr
			  (handle, nnz, csrVal_tmp,
			   csrVal, perm,
			   rocsparse_index_base_zero));
      //      BML_CHECK_ROCSPARSE(rocsparse_spmat_set_values
      //			  ((rocsparse_mat_descr) matC_tmp, csrValC_tmp_sorted));
    }

#pragma omp target exit data map(delete:csrVal_tmp[:nnz],perm[:nnz],dwork[:lworkInBytes])
    free(csrVal_tmp);
    free(perm);
    free(dwork);

    BML_CHECK_ROCSPARSE(rocsparse_destroy_mat_descr(mat));
}

/** Prune (threshold) a rocsparse CSR matrix
 *
 *
 *  \ingroup
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
void TYPED_FUNC(
    bml_prune_rocsparse_ellpack) (
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A,
    double threshold_in)
{
    int N = A->N;
    int M = A->M;
    int *csrColInd = A->csrColInd;
    int *csrRowPtr = A->csrRowPtr;
    REAL_T *csrVal = A->csrVal;
    int nnz;

    int i;
    
    int *csrRowPtr_tmp;
    int *csrColInd_tmp;
    REAL_T *csrVal_tmp;
    int nnz_tmp;
    
    rocsparse_mat_descr mat, mat_tmp;
      
    size_t lworkInBytes = 0;
    char *dwork = NULL;

    REAL_T threshold = (REAL_T)threshold_in;

#pragma omp target map(from:nnz)
    {
      nnz = csrRowPtr[N];
    }
    
    // Temporary array for sorting

    csrRowPtr_tmp = (int *) malloc(sizeof(int)*(N + 1));
    csrColInd_tmp = (int *) malloc(sizeof(int)* nnz);
    csrVal_tmp = (REAL_T *) malloc(sizeof(REAL_T)* nnz);
    nnz_tmp = nnz;

#pragma omp target enter data map(alloc:csrRowPtr_tmp[:N+1],csrColInd_tmp[:nnz],csrVal_tmp[:nnz])
    
#pragma omp target teams distribute parallel for
    for (i=0; i<=N; i++) {
      csrRowPtr_tmp[i] = csrRowPtr[i];
    }
    
#pragma omp target teams distribute parallel for
    for (i=0; i<nnz; i++) {
      csrColInd_tmp[i] = csrColInd[i];
      csrVal_tmp[i] = ((REAL_T *)csrVal)[i];
    }
        // rocSPARSE APIs
    rocsparse_status status = rocsparse_status_success;

    rocsparse_datatype computeType = BML_ROCSPARSE_T;

    BML_CHECK_ROCSPARSE(rocsparse_create_mat_descr
			(&mat));
    BML_CHECK_ROCSPARSE(rocsparse_create_mat_descr
			(&mat_tmp));

    // Prune the output matrix and overwrite the A matrix with the result
    
    // xprune stage 1 = determine working buffer size
#pragma omp target data use_device_ptr(csrVal_tmp,csrRowPtr_tmp, csrColInd_tmp,csrVal, csrRowPtr, csrColInd)
    {
      BML_CHECK_ROCSPARSE(bml_rocsparse_xprune_csr2csr_buffer_size
			  (handle, N, N, nnz_tmp,
			   mat_tmp, csrVal_tmp,
			   csrRowPtr_tmp, csrColInd_tmp, &threshold,
			   mat, csrVal,
			   csrRowPtr, csrColInd, &lworkInBytes));
    }
    // Allocate the working buffer on the host and device
    dwork = (char *) malloc(sizeof(char) * lworkInBytes);
#pragma omp target enter data map(alloc:dwork[:lworkInBytes])
    
    // xprune stages 2, 3 = determine nnz, perform pruning
#pragma omp target data use_device_ptr(csrVal_tmp,csrRowPtr_tmp, csrColInd_tmp,dwork,csrVal, csrRowPtr, csrColInd)
    {
      BML_CHECK_ROCSPARSE(bml_rocsparse_xprune_csr2csr_nnz
			  (handle, N, N, nnz_tmp,
			   mat_tmp, csrVal_tmp,
			   csrRowPtr_tmp, csrColInd_tmp, &threshold,
			   mat, csrRowPtr,
			   &nnz, dwork));
      BML_CHECK_ROCSPARSE(bml_rocsparse_xprune_csr2csr
			  (handle, N, N, nnz_tmp,
			   (rocsparse_mat_descr) mat_tmp, csrVal_tmp,
			   csrRowPtr_tmp, csrColInd_tmp, &threshold,
			   (rocsparse_mat_descr) mat, csrVal,
			   csrRowPtr, csrColInd, dwork));
    }


#pragma omp target exit data map(delete:csrRowPtr_tmp[:N+1],csrColInd_tmp[:nnz_tmp],csrVal_tmp[:nnz_tmp],dwork[:lworkInBytes])
    free(csrRowPtr_tmp);
    free(csrColInd_tmp);
    free(csrVal_tmp);
    free(dwork);
    
    BML_CHECK_ROCSPARSE(rocsparse_destroy_mat_descr(mat));
    BML_CHECK_ROCSPARSE(rocsparse_destroy_mat_descr(mat_tmp));
}
#endif
#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE)
/** Ellpack to cuCSR conversion.
 *
 *  Convert from Ellpack format to cusparse csr format.
 *  Naive implementation for testing. Use thrust library for optimal code.
 *
 *  \ingroup
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
void TYPED_FUNC(
    bml_ellpack2cucsr_ellpack) (
    bml_matrix_ellpack_t * A)
{
    int A_N = A->N;
    int A_M = A->M;
    int *A_index = A->index;
    int *csrColInd = A->csrColInd;
    int *A_nnz = A->nnz;
    int *csrRowPtr = A->csrRowPtr;
    REAL_T *A_value = A->value;
    REAL_T *csrVal = A->csrVal;

#pragma omp target update from(A_nnz[:A_N])

    csrRowPtr[0] = 0;
    for (int i = 0; i < A_N; i++)
    {
        csrRowPtr[i + 1] = csrRowPtr[i] + A_nnz[i];
    }
#pragma omp target update to(csrRowPtr[:A_N+1])

#pragma omp target teams distribute parallel for \
    shared(A_N, A_M, A_nnz, A_index, A_value) \
    shared(csrVal, csrRowPtr, csrColInd)
    for (int i = 0; i < A_N; i++)
    {
        for (int j = 0; j < A_nnz[i]; j++)
        {
            int idx = csrRowPtr[i] + j;
            csrVal[idx] = A_value[ROWMAJOR(i, j, A_N, A_M)];
            csrColInd[idx] = A_index[ROWMAJOR(i, j, A_N, A_M)];
        }
    }
}

/** cuCSR to Ellpack conversion.
 *
 *  Convert from cusparse csr format to Ellpack format.
 *  Naive implementation for testing. Use thrust library for optimal code.
 *
 *  \ingroup
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */

void TYPED_FUNC(
    bml_cucsr2ellpack_ellpack) (
    bml_matrix_ellpack_t * A)
{
    int A_N = A->N;
    int A_M = A->M;
    int *A_index = A->index;
    int *csrColInd = A->csrColInd;
    int *A_nnz = A->nnz;

    int *csrRowPtr = A->csrRowPtr;
    REAL_T *A_value = A->value;
    REAL_T *csrVal = A->csrVal;

#pragma omp target teams distribute parallel for \
    shared(A_N, A_M, A_nnz, A_index, A_value) \
    shared(csrVal, csrRowPtr, csrColInd)
    for (int i = 0; i < A_N; i++)
    {
        A_nnz[i] = csrRowPtr[i + 1] - csrRowPtr[i];
        for (int j = 0; j < A_nnz[i]; j++)
        {
            int idx = csrRowPtr[i] + j;
            A_value[ROWMAJOR(i, j, A_N, A_M)] = csrVal[idx];
            A_index[ROWMAJOR(i, j, A_N, A_M)] = csrColInd[idx];
        }
    }
}
#endif
