#include "bml_allocate.h"
#include "bml_logger.h"
#include "dense/bml_allocate_dense.h"

#include <stdlib.h>

/** Allocate and zero a chunk of memory.
 *
 * \param size The size of the memory.
 * \return A pointer to the allocated chunk.
 */
void *bml_allocate_memory(const size_t size)
{
    void *ptr = calloc(1, size);
    if(ptr == NULL) {
        bml_log(BML_LOG_ERROR, "error allocating memory\n");
    }
    return ptr;
}

/** Deallocate a chunk of memory.
 *
 * \param ptr A pointer to the previously allocated chunk.
 */
void bml_free_memory(void *ptr)
{
    free(ptr);
}

/** Allocate a matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. The
 *  function does not check whether the matrix is already allocated.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_type The matrix type.
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param A The matrix.
 *  \param N The matrix size.
 */
void bml_allocate(const bml_matrix_type_t matrix_type,
                  const bml_matrix_precision_t matrix_precision,
                  bml_matrix_t *A,
                  const int N)
{
    switch(matrix_type) {
    case dense:
        bml_allocate_dense(matrix_precision, A, N);
    default:
        bml_log(BML_LOG_ERROR, "unknown matrix type\n");
    }
}

/** Deallocate a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void bml_deallocate(bml_matrix_t **A)
{
    bml_matrix_type_t *matrix_type = *A;

    if(*A != NULL) {
        switch(*matrix_type) {
        case dense:
            bml_deallocate_dense(*A);
        default:
            bml_log(BML_LOG_ERROR, "unknown matrix type\n");
        }
    }
    A = NULL;
}
