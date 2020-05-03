#include "../../macros.h"
#include "../../typed.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_utilities.h"
#include "bml_types_csr.h"
#include "bml_setters_csr.h"
#include "bml_utilities_csr.h"

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
    bml_read_bml_matrix_csr) (
    bml_matrix_csr_t * A,
    char *filename)
{
    FILE *hFile;
    char header1[20], header2[20], header3[20], header4[20], header5[20];
    int hdimx, nnz, irow, icol;
    REAL_T val;

    hFile = fopen(filename, "r");

    // Read header
    if (fscanf(hFile, "%s %s %s %s %s", header1, header2, header3, header4,
               header5) != 5)
    {
        LOG_ERROR("read error on header\n");
    }

    int symflag = strcmp(header5, "symmetric");

    // Read N, N, # of non-zeroes
    if (fscanf(hFile, "%d %d %d", &hdimx, &hdimx, &nnz) != 3)
    {
        LOG_ERROR("read error\n");
    }

    char *FMT = "";
    switch (A->matrix_precision)
    {
        case single_real:
            FMT = "%d %d %g\n";
            break;
        case double_real:
            FMT = "%d %d %lg\n";
            break;
        case single_complex:
            FMT = "%d %d %g\n";
            break;
        case double_complex:
            FMT = "%d %d %lg\n";
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }

    // Read in values
    for (int i = 0; i < nnz; i++)
    {
        if (fscanf(hFile, FMT, &irow, &icol, &val) != 3)
        {
            LOG_ERROR("read error\n");
        }
        irow--;
        icol--;
        // set new element
        TYPED_FUNC(csr_set_row_element_new) (A->data_[irow], icol, &val);
        // Set symmetric value if necessary
        if (symflag == 0 && icol != irow)
        {
            TYPED_FUNC(csr_set_row_element_new) (A->data_[icol], irow, &val);
        }
    }

    fclose(hFile);
}

/** Write a Matrix Market format file from a bml matrix.
 *
 *  \ingroup utilities_group
 *
 *  \param A The matrix to be written
 *  \param filename The Matrix Market format file
 */
void TYPED_FUNC(
    bml_write_bml_matrix_csr) (
    bml_matrix_csr_t * A,
    char *filename)
{
    FILE *mFile;
    int msum;

    int N = A->N_;

    REAL_T *vals = NULL;
    int *cols = NULL;

    // Only write from rank 0
    if (bml_printRank() != 1)
        return;

    mFile = fopen(filename, "w");

    // Write header
    fprintf(mFile, "%%%%%%MatrixMarket matrix coordinate real general\n");

    // Collect number of non-zero elements
    // Write out matrix size as dense and number of non-zero elements
    msum = 0;
    for (int i = 0; i < N; i++)
    {
        msum += A->data_[i]->NNZ_;
    }
    fprintf(mFile, "%d %d %d\n", N, N, msum);

    // Write out non-zero elements
    for (int i = 0; i < N; i++)
    {
        cols = A->data_[i]->cols_;
        vals = (REAL_T *) A->data_[i]->vals_;
        const int annz = A->data_[i]->NNZ_;
        for (int pos = 0; pos < annz; pos++)
        {
            fprintf(mFile, "%d %d %20.15e\n", i + 1,
                    cols[pos] + 1, REAL_PART(vals[pos]));
        }
    }

    fclose(mFile);
}
