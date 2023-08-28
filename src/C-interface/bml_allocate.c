#include "bml_allocate.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_parallel.h"
#include "dense/bml_allocate_dense.h"
#include "ellpack/bml_allocate_ellpack.h"
#include "ellblock/bml_allocate_ellblock.h"
#include "csr/bml_allocate_csr.h"
#ifdef BML_USE_MPI
#include "distributed2d/bml_allocate_distributed2d.h"
#endif

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Check if matrix is allocated.
 *
 * \ingroup allocate_group_C
 *
 * \param A[in,out] Matrix
 * \return \f$ > 0 \f$ if allocated, else -1
 */
int
bml_allocated(
    bml_matrix_t * A)
{
    return bml_get_N(A);
}

/** Allocate and zero a chunk of memory.
 *
 * \ingroup allocate_group_C
 *
 * \param size The size of the memory.
 * \return A pointer to the allocated chunk.
 */
void *
bml_allocate_memory(
    size_t size)
{
#if defined(INTEL_OPT)
    char *ptr = _mm_malloc(size, MALLOC_ALIGNMENT);
#pragma omp parallel for simd
#pragma vector aligned
    for (size_t i = 0; i < size; i++)
    {
        __assume_aligned(ptr, MALLOC_ALIGNMENT);
        ptr[i] = 0;
    }
#elif defined(HAVE_POSIX_MEMALIGN)
    char *ptr;
    posix_memalign((void **) &ptr, MALLOC_ALIGNMENT, size);
#pragma omp simd
    for (size_t i = 0; i < size; i++)
    {
        ptr[i] = 0;
    }
#else
    void *ptr = calloc(1, size);
#endif

    if (ptr == NULL)
    {
        LOG_ERROR("error allocating memory of size %d: %s\n", size,
                  strerror(errno));
    }
    return (void *) ptr;
}

/** Allocate a chunk of memory without initialization.
 *
 * \ingroup allocate_group_C
 *
 * \param size The size of the memory.
 * \return A pointer to the allocated chunk.
 */
void *
bml_noinit_allocate_memory(
    size_t size)
{
#if defined(INTEL_OPT)
    void *ptr = _mm_malloc(size, MALLOC_ALIGNMENT);
#elif defined(HAVE_POSIX_MEMALIGN)
    void *ptr;
    posix_memalign(&ptr, MALLOC_ALIGNMENT, size);
#else
    void *ptr = malloc(size);
#endif
    if (ptr == NULL)
    {
        LOG_ERROR("error allocating memory: %s\n", strerror(errno));
    }
    return ptr;
}

/** Reallocate a chunk of memory.
 *
 * \ingroup allocate_group_C
 *
 * \param size The size of the memory.
 * \return A pointer to the reallocated chunk.
 */
void *
bml_reallocate_memory(
    void *ptr,
    const size_t size)
{
    void *ptr_new = realloc(ptr, size);
    if (ptr_new == NULL)
    {
        LOG_ERROR("error reallocating memory: %s\n", strerror(errno));
    }
    return ptr_new;
}

/** Deallocate a chunk of memory.
 *
 * \ingroup allocate_group_C
 *
 * \param ptr A pointer to the previously allocated chunk.
 */
void
bml_free_memory(
    void *ptr)
{
#ifdef INTEL_OPT
    _mm_free(ptr);
#else
    free(ptr);
#endif
}

/** De-allocate a chunk of memory that was allocated inside a C
 * function. This is used by the Fortran bml_free_C interface. Note
 * the "pointer to pointer" in the API.
 *
 * \ingroup allocate_group_C
 *
 * \param ptr A pointer to the previously allocated chunk.
 */
void
bml_free_ptr(
    void **ptr)
{
    bml_free_memory(*ptr);
}

/** Deallocate a matrix.
 *
 * \ingroup allocate_group_C
 *
 * \param A[in,out] The matrix.
 */
void
bml_deallocate(
    bml_matrix_t ** A)
{
    if (A == NULL)
    {
        LOG_DEBUG("A is NULL\n");
    }
    else if (*A == NULL)
    {
        LOG_DEBUG("*A is NULL\n");
    }
    else
    {
        LOG_DEBUG("deallocating bml matrix\n");
        switch (bml_get_type(*A))
        {
            case dense:
                bml_deallocate_dense(*A);
                break;
            case ellpack:
                bml_deallocate_ellpack(*A);
                break;
            case ellblock:
                bml_deallocate_ellblock(*A);
                break;
            case csr:
                bml_deallocate_csr(*A);
                break;
#ifdef BML_USE_MPI
            case distributed2d:
                bml_deallocate_distributed2d(*A);
                break;
#endif
            default:
                LOG_ERROR("unknown matrix type (%d)\n", bml_get_type(*A));
                break;
        }
        *A = NULL;
    }
}

/** Clear a matrix.
 *
 * \ingroup allocate_group_C
 *
 * \param A[in,out] The matrix.
 */
void
bml_clear(
    bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_clear_dense(A);
            break;
        case ellpack:
            bml_clear_ellpack(A);
            break;
        case ellblock:
            bml_clear_ellblock(A);
            break;
        case csr:
            bml_clear_csr(A);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            bml_clear_distributed2d(A);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type (%d)\n", bml_get_type(A));
            break;
    }
}

/** Allocate a matrix without initializing.
 *
 *  Note that the matrix \f$ A \f$ will be newly allocated. The
 *  function does not check whether the matrix is already allocated.
 *
 *  \ingroup allocate_group_C
 *
 *  \param matrix_type The matrix type.
 *  \param matrix_precision The precision of the matrix.
 *  \param matrix_dimension The matrix size.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_t *
bml_noinit_rectangular_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    bml_matrix_dimension_t matrix_dimension,
    bml_distribution_mode_t distrib_mode)
{
    LOG_DEBUG("noinit matrix of size %d (or zero matrix for dense)\n",
              matrix_dimension.N_rows);
#ifdef BML_USE_MPI
    if (distrib_mode == distributed)
        return bml_noinit_matrix_distributed2d(matrix_type, matrix_precision,
                                               matrix_dimension.N_rows,
                                               matrix_dimension.N_nz_max);
    else
#endif
        switch (matrix_type)
        {
            case dense:
                return bml_noinit_matrix_dense(matrix_precision,
                                               matrix_dimension,
                                               distrib_mode);
                break;
            case ellpack:
                return bml_noinit_matrix_ellpack(matrix_precision,
                                                 matrix_dimension,
                                                 distrib_mode);
                break;
            case ellblock:
                return bml_noinit_matrix_ellblock(matrix_precision,
                                                  matrix_dimension,
                                                  distrib_mode);
                break;
            case csr:
                return bml_noinit_matrix_csr(matrix_precision,
                                             matrix_dimension, distrib_mode);
                break;
            default:
                LOG_ERROR("unknown matrix type\n");
                break;
        }
    return NULL;
}

/** Allocate a block matrix
 *
 * \param matrix_type The matrix type.
 * \param matrix_precision The precision of the matrix.
 * \param NB The number of blocks in a row.
 * \param bsizes The sizes of each block
 * \param MB The number of non-zeroes blocks per row.
 * \param distrib_mode The distribution mode.
 * \return The matrix.
 */
bml_matrix_t *
bml_block_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int NB,
    int MB,
    int M,
    int *bsizes,
    bml_distribution_mode_t distrib_mode)
{
    LOG_DEBUG("block matrix with %d blocks\n", NB);
    switch (matrix_type)
    {
        case ellpack:
            return bml_block_matrix_ellblock(matrix_precision,
                                             NB, MB, M, bsizes, distrib_mode);
            break;
        case ellblock:
            return bml_block_matrix_ellblock(matrix_precision,
                                             NB, MB, M, bsizes, distrib_mode);
            break;
        default:
            LOG_ERROR("unsupported matrix type (type ID %d)\n", matrix_type);
            break;
    }
    return NULL;
}

/** Allocate a matrix without initializing.
 *
 *  Note that the matrix \f$ A \f$ will be newly allocated. The
 *  function does not check whether the matrix is already allocated.
 *
 *  \ingroup allocate_group_C
 *
 *  \param matrix_type The matrix type.
 *  \param matrix_precision The precision of the matrix.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_t *
bml_noinit_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    bml_matrix_dimension_t matrix_dimension = { N, N, M };
    return bml_noinit_rectangular_matrix(matrix_type, matrix_precision,
                                         matrix_dimension, distrib_mode);
}

/** Allocate the zero matrix.
 *
 *  Note that the matrix \f$ A \f$ will be newly allocated. The
 *  function does not check whether the matrix is already allocated.
 *
 *  \ingroup allocate_group_C
 *
 *  \param matrix_type The matrix type.
 *  \param matrix_precision The precision of the matrix.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_t *
bml_zero_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    LOG_DEBUG("zero matrix of size %d\n", N);

#ifdef BML_USE_MPI
    if (distrib_mode == distributed)
        return bml_zero_matrix_distributed2d(matrix_type, matrix_precision, N,
                                             M);
    else
#endif
    {
        bml_matrix_dimension_t matrix_dimension = { N, N, M };
        switch (matrix_type)
        {
            case dense:
                return bml_zero_matrix_dense(matrix_precision,
                                             matrix_dimension, distrib_mode);
                break;
            case ellpack:
                return bml_zero_matrix_ellpack(matrix_precision, N, M,
                                               distrib_mode);
                break;
            case ellblock:
                return bml_zero_matrix_ellblock(matrix_precision, N, M,
                                                distrib_mode);
                break;
            case csr:
                return bml_zero_matrix_csr(matrix_precision, N, M,
                                           distrib_mode);
                break;
            default:
                LOG_ERROR("unknown matrix type\n");
                break;
        }
    }
    return NULL;
}

/** Allocate a random matrix.
 *
 *  Note that the matrix \f$ A \f$ will be newly allocated. The
 *  function does not check whether the matrix is already allocated.
 *
 *  \ingroup allocate_group_C
 *
 *  \param matrix_type The matrix type.
 *  \param matrix_precision The precision of the matrix.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_t *
bml_random_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
#ifdef BML_USE_MPI
    if (distrib_mode == distributed)
        return bml_random_matrix_distributed2d(matrix_type, matrix_precision,
                                               N, M);
    else
#endif
        switch (matrix_type)
        {
            case dense:
                return bml_random_matrix_dense(matrix_precision, N,
                                               distrib_mode);
                break;
            case ellpack:
                return bml_random_matrix_ellpack(matrix_precision, N, M,
                                                 distrib_mode);
                break;
            case ellblock:
                return bml_random_matrix_ellblock(matrix_precision, N, M,
                                                  distrib_mode);
                break;
            case csr:
                return bml_random_matrix_csr(matrix_precision, N, M,
                                             distrib_mode);
                break;
            default:
                LOG_ERROR("unknown matrix type (type ID %d)\n", matrix_type);
                break;
        }
    return NULL;
}

/** Allocate a banded matrix.
 *
 *  Note that the matrix \f$ A \f$ will be newly allocated. The
 *  function does not check whether the matrix is already allocated.
 *
 *  \ingroup allocate_group_C
 *
 *  \param matrix_type The matrix type.
 *  \param matrix_precision The precision of the matrix.
 *  \param N The matrix size.
 *  \param M The bandwidth of the matrix.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_t *
bml_banded_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    LOG_DEBUG("banded matrix of size %d\n", N);
    switch (matrix_type)
    {
        case dense:
            return bml_banded_matrix_dense(matrix_precision, N, M,
                                           distrib_mode);
            break;
        case ellpack:
            return bml_banded_matrix_ellpack(matrix_precision, N, M,
                                             distrib_mode);
            break;
        case ellblock:
            return bml_banded_matrix_ellblock(matrix_precision, N, M,
                                              distrib_mode);
            break;
        case csr:
            return bml_banded_matrix_csr(matrix_precision, N, M,
                                         distrib_mode);
            break;
        default:
            LOG_ERROR("unknown matrix type (type ID %d)\n", matrix_type);
            break;
    }
    return NULL;
}

/** Allocate the identity matrix.
 *
 *  Note that the matrix \f$ A \f$ will be newly allocated. The
 *  function does not check whether the matrix is already allocated.
 *
 *  \ingroup allocate_group_C
 *
 *  \param matrix_type The matrix type.
 *  \param matrix_precision The precision of the matrix.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_t *
bml_identity_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    LOG_DEBUG("identity matrix of size %d\n", N);
#ifdef BML_USE_MPI
    if (distrib_mode == distributed)
        return bml_identity_matrix_distributed2d(matrix_type,
                                                 matrix_precision, N, M);
    else
#endif
        switch (matrix_type)
        {
            case dense:
                return bml_identity_matrix_dense(matrix_precision, N,
                                                 distrib_mode);
                break;
            case ellpack:
                return bml_identity_matrix_ellpack(matrix_precision, N, M,
                                                   distrib_mode);
                break;
            case ellblock:
                return bml_identity_matrix_ellblock(matrix_precision, N, M,
                                                    distrib_mode);
                break;
            case csr:
                return bml_identity_matrix_csr(matrix_precision, N, M,
                                               distrib_mode);
                break;
            default:
                LOG_ERROR("unknown matrix type (type ID %d)\n", matrix_type);
                break;
        }
    return NULL;
}

/** Update a domain for a bml matrix.
 *
 * \ingroup allocate_group_C
 *
 * \param A Matrix with domain
 * \param localPartMin First part on each rank
 * \param localPartMax Last part on each rank
 * \param nnodesInPart Number of nodes in each part
 */
void
bml_update_domain_matrix(
    bml_matrix_t * A,
    int *localPartMin,
    int *localPartMax,
    int *nnodesInPart)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_update_domain_dense(A, localPartMin, localPartMax,
                                    nnodesInPart);
            break;
        case ellpack:
            bml_update_domain_ellpack(A, localPartMin, localPartMax,
                                      nnodesInPart);
            break;
        default:
            LOG_ERROR("unknown matrix type (%d)\n", bml_get_type(A));
            break;
    }
}
