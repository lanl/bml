#ifdef BML_USE_MAGMA
#include <stdbool.h>
#include "magma_v2.h"
#endif

#include "../../typed.h"
#include "../../macros.h"
#include "../bml_logger.h"
#include "../bml_utilities.h"
#include "../bml_allocate.h"
#include "bml_types_dense.h"
#include "bml_utilities_dense.h"
#include "bml_export_dense.h"
#include "bml_allocate_dense.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/** Read in a bml matrix from Matrix Market format.
 *
 *  \ingroup utilities_group
 *
 *  \param A The matrix to be read
 *  \param filename The Matrix Market format file
 */
void TYPED_FUNC(
    bml_read_bml_matrix_dense) (
    bml_matrix_dense_t * A,
    char *filename)
{
    FILE *matrix_file;
    char header1[20], header2[20], header3[20], header4[20], header5[20];
    int hdimx, nnz, irow, icol;
    int values_read;
    double real_part;
#if (defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX))
    double imaginary_part;
#endif

    int N = A->N;

#ifdef BML_USE_MAGMA
    REAL_T *A_matrix = bml_allocate_memory(N * N * sizeof(REAL_T));
#else
    REAL_T *A_matrix = A->matrix;
#endif

    matrix_file = fopen(filename, "r");

    // Read header
    if ((values_read = fscanf
         (matrix_file, "%s %s %s %s %s", header1, header2, header3, header4,
          header5)) < 5)
    {
        LOG_ERROR("Line 1, expected 5 entries, read %d\n", values_read);
    }

    // Read N, N, # of non-zeroes
    if ((values_read =
         fscanf(matrix_file, "%d %d %d", &hdimx, &hdimx, &nnz)) < 3)
    {
        LOG_ERROR("Line 2, Expected 3 entries, read %d\n", values_read);
    }

    // Read in values
    for (int i = 0; i < nnz; i++)
    {
#if defined(SINGLE_REAL)
        if ((values_read =
             fscanf(matrix_file, "%d %d %lg\n", &irow, &icol,
                    &real_part)) < 3)
        {
            LOG_ERROR("Line %d, expected 3 entries, read %d\n", i + 3,
                      values_read);
        }
        A_matrix[ROWMAJOR(irow - 1, icol - 1, N, N)] = real_part;
#elif defined(DOUBLE_REAL)
        if ((values_read =
             fscanf(matrix_file, "%d %d %lg\n", &irow, &icol,
                    &real_part)) < 3)
        {
            LOG_ERROR("Line %d, expected 3 entries, read %d\n", i + 3,
                      values_read);
        }
        A_matrix[ROWMAJOR(irow - 1, icol - 1, N, N)] = real_part;
#elif defined(SINGLE_COMPLEX)
        if ((values_read =
             fscanf(matrix_file, "%d %d %lg %lg\n", &irow, &icol, &real_part,
                    &imaginary_part)) < 4)
        {
            LOG_ERROR("Line %d, expected 4 entries, read %d\n", i + 3,
                      values_read);
        }
        A_matrix[ROWMAJOR(irow - 1, icol - 1, N, N)] =
            real_part + I * imaginary_part;
#elif defined(DOUBLE_COMPLEX)
        if ((values_read =
             fscanf(matrix_file, "%d %d %lg %lg\n", &irow, &icol, &real_part,
                    &imaginary_part)) < 4)
        {
            LOG_ERROR("Line %d, expected 4 entries, read %d\n", i + 3,
                      values_read);
        }
        A_matrix[ROWMAJOR(irow - 1, icol - 1, N, N)] =
            real_part + I * imaginary_part;
#else
        LOG_ERROR("unknown precision\n");
#endif
    }

#ifdef BML_USE_MAGMA
    MAGMA(setmatrix) (N, N, (MAGMA_T *) A_matrix, N, A->matrix, A->ld,
                      bml_queue());
    bml_free_memory(A_matrix);
#endif
#ifdef MKL_GPU
// push back to GPU
#pragma omp target update to(A_matrix[0:N*N])
#endif
    fclose(matrix_file);
}

/** Write a Matrix Market format file from a bml matrix.
 *
 *  \ingroup utilities_group
 *
 * Note, all matrix elements are written, even the ones that are zero.
 *
 * \param A The matrix to be written
 * \param filename The Matrix Market format file
 */
void TYPED_FUNC(
    bml_write_bml_matrix_dense) (
    bml_matrix_dense_t * A,
    char *filename)
{
    FILE *matrix_file;

    int N = A->N;
    int msum = 0;

#ifdef BML_USE_MAGMA
    REAL_T *A_matrix = bml_noinit_allocate_memory(N * N * sizeof(REAL_T));
    MAGMA(getmatrix) (N, N, A->matrix, A->ld, (MAGMA_T *) A_matrix, N,
                      bml_queue());
#else
    REAL_T *A_matrix = A->matrix;
#ifdef MKL_GPU
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif

#endif

    matrix_file = fopen(filename, "w");

    // Write header
#if defined(SINGLE_REAL) || defined(DOUBLE_REAL)
    fprintf(matrix_file,
            "%%%%%%MatrixMarket matrix coordinate real general\n");
#elif defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
    fprintf(matrix_file,
            "%%%%%%MatrixMarket matrix coordinate complex general\n");
#endif

    // count number of non-zero elements
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            REAL_T value = A_matrix[ROWMAJOR(i, j, N, N)];
            if (ABS(value) > 0.)
                msum++;
        }
    }

    // Write out matrix size as dense and number of non-zero elements
    fprintf(matrix_file, "%d %d %d\n", N, N, msum);

    // Write out non-zero elements
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            REAL_T value = A_matrix[ROWMAJOR(i, j, N, N)];
            if (ABS(value) > 0.)
            {
#if defined(SINGLE_REAL) || defined(DOUBLE_REAL)
                fprintf(matrix_file, "%d %d %20.15g\n", i + 1, j + 1, value);
#elif defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
                fprintf(matrix_file, "%d %d %20.15g %20.15g\n", i + 1, j + 1,
                        REAL_PART(value), IMAGINARY_PART(value));
#else
                LOG_ERROR("unknown precision\n");
#endif
            }
        }
    }

#ifdef BML_USE_MAGMA
    bml_free_memory(A_matrix);
#endif
    fclose(matrix_file);
}

void TYPED_FUNC(
    bml_print_bml_matrix_dense) (
    bml_matrix_dense_t * A,
    int i_l,
    int i_u,
    int j_l,
    int j_u)
{
#ifdef BML_USE_MAGMA
    //copy matrix data from GPU to CPU so that we can use
    //bml_print_dense_matrix function with lower/upper row/column
    //index
    REAL_T *A_matrix = bml_allocate_memory(sizeof(REAL_T) * A->N * A->N);
    MAGMA(getmatrix) (A->N, A->N,
                      A->matrix, A->ld, (MAGMA_T *) A_matrix, A->N,
                      bml_queue());
#else
    REAL_T *A_matrix = (REAL_T *) A->matrix;
    int N = A->N;

#ifdef MKL_GPU
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif

#endif

    bml_print_dense_matrix(A->N, A->matrix_precision, dense_row_major,
                           A_matrix, i_l, i_u, j_l, j_u);

#ifdef BML_USE_MAGMA
    bml_free_memory(A_matrix);
#endif
}
