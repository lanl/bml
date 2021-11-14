#include "bml_allocate_distributed2d.h"
#include "../bml_introspection.h"
#include "../bml_allocate.h"
#include "../bml_convert.h"
#include "bml_copy_distributed2d.h"
#include "bml_diagonalize_distributed2d.h"
#include "../bml_diagonalize.h"
#include "../bml_logger.h"
#include "bml_types_distributed2d.h"
#include "../bml_types.h"
#include "../bml_utilities.h"
#include "../../macros.h"
#include "../../typed.h"
#include "../bml_transpose.h"
#include "../bml_copy.h"

#include <mpi.h>

#include <complex.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef BML_USE_SCALAPACK
// since scalapack does not provide an include file, we add prototypes here
int Csys2blacs_handle(
    MPI_Comm);
void Cblacs_gridinfo(
    int,
    int *,
    int *,
    int *,
    int *);
void Cblacs_gridinit(
    int *,
    char[],
    int,
    int);
int NUMROC(
    int *,
    int *,
    int *,
    int *,
    int *);

#if defined(SINGLE_REAL)
void PSSYEVD(
    const char *const,
    const char *const,
    int *,
    REAL_T *,
    int *,
    int *,
    int *,
    REAL_T *,
    REAL_T *,
    int *,
    int *,
    int *,
    REAL_T *,
    int *,
    int *,
    int *,
    int *);
#define SYEVD PSSYEVD
#elif defined(DOUBLE_REAL)
void PDSYEVD(
    const char *const,
    const char *const,
    int *,
    double *,
    int *,
    int *,
    int *,
    double *,
    double *,
    int *,
    int *,
    int *,
    double *,
    int *,
    int *,
    int *,
    int *);
#define SYEVD PDSYEVD
#elif defined(SINGLE_COMPLEX)
void PCHEEVD(
    const char *const,
    const char *const,
    int *,
    REAL_T *,
    int *,
    int *,
    int *,
    float *,
    REAL_T *,
    int *,
    int *,
    int *,
    REAL_T *,
    int *,
    float *,
    int *,
    int *,
    int *,
    int *);
#define SYEVD PCHEEVD
#elif defined(DOUBLE_COMPLEX)
void PZHEEVD(
    const char *const,
    const char *const,
    int *,
    REAL_T *,
    int *,
    int *,
    int *,
    double *,
    REAL_T *,
    int *,
    int *,
    int *,
    REAL_T *,
    int *,
    double *,
    int *,
    int *,
    int *,
    int *);
#define SYEVD PZHEEVD
#endif
#endif // BML_USE_SCALAPACK

/** Diagonalize matrix.
 *
 *  \ingroup diag_group
 *
 *  \param A The matrix A
 *  \param eigenvalues Eigenvalues of A
 *  \param eigenvectors Eigenvectors of A
 */
void TYPED_FUNC(
    bml_diagonalize_distributed2d) (
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors)
{
#ifdef BML_USE_SCALAPACK
    REAL_T *typed_eigenvalues = (REAL_T *) eigenvalues;
    // distributed2d format uses a row block distribution
    char order = 'R';
    int np_rows = A->nprows;
    int np_cols = A->npcols;
    int my_prow = A->myprow;
    int my_pcol = A->mypcol;
    int my_blacs_ctxt = Csys2blacs_handle(A->comm);
    Cblacs_gridinit(&my_blacs_ctxt, &order, np_rows, np_cols);
    Cblacs_gridinfo(my_blacs_ctxt, &np_rows, &np_cols, &my_prow, &my_pcol);

    int m = A->N;
    int mb = A->N / np_rows;

    int desc[9];
    desc[0] = 1;
    desc[1] = my_blacs_ctxt;
    desc[2] = A->N;
    desc[3] = A->N;
    desc[4] = mb;
    desc[5] = mb;
    desc[6] = 0;
    desc[7] = 0;
    desc[8] = A->N / np_rows;

    bml_matrix_t *Alocal = A->matrix;

    // copy matrix into tmp array since scalapack will destroy its content
    bml_matrix_t *zmat = NULL;
    bml_matrix_t *amat = NULL;
    if (bml_get_type(Alocal) == dense)
    {
        amat = bml_copy_new(Alocal);
        zmat = eigenvectors->matrix;
    }
    else
    {
        LOG_INFO("WARNING: convert local matrices to dense...\n");
        // convert local matrix to dense
        amat = bml_convert(Alocal, dense, A->matrix_precision,
                           -1, sequential);
        zmat = bml_convert(eigenvectors->matrix, dense, A->matrix_precision,
                           -1, sequential);
    }

    // transpose to satisfy column major ScaLapack convention
    // (global matrix assumed symmetric, so no need for communications)
    if (A->myprow != A->mypcol)
        bml_transpose(amat);

    REAL_T *pzmat = bml_get_data_ptr(zmat);
    assert(pzmat != NULL);
    REAL_T *atmp = bml_get_data_ptr(amat);
    assert(atmp != NULL);

    int ione = 1;
    int izero = 0;
    int np0 = NUMROC(&m, &mb, &my_prow, &izero, &np_rows);
    int nq0 = NUMROC(&m, &mb, &my_pcol, &izero, &np_cols);
#if defined(SINGLE_REAL) || defined(DOUBLE_REAL)
    int lwork = MAX(1 + 6 * m + 2 * np0 * nq0,
                    3 * m + MAX(mb * (np0 + 1), 3 * mb)) + 2 * m;
#else
    int lwork = m + (2 * np0 + mb) * mb;
#endif
#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
    float *ev = malloc(A->N * sizeof(float));
#endif
#if defined(DOUBLE_REAL) || defined(DOUBLE_COMPLEX)
    double *ev = malloc(A->N * sizeof(double));
#endif

#if defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
    int np = NUMROC(&m, &mb, &my_prow, &ione, &np_rows);
    int nq = NUMROC(&m, &mb, &my_pcol, &ione, &np_cols);
    int lrwork = 1 + 9 * m + 3 * np * nq;
#endif

#if defined(SINGLE_COMPLEX)
    float *rwork = bml_allocate_memory(lrwork * sizeof(float));
#endif
#if defined(DOUBLE_COMPLEX)
    double *rwork = bml_allocate_memory(lrwork * sizeof(double));
#endif

    int liwork = 7 * m + 8 * np_cols + 2;
    int *iwork = bml_allocate_memory(liwork * sizeof(int));
    // Solve matrix eigenvalue problem problem
    char jobz = 'V';
    char uplo = 'U';
    int info;
    //LOG_INFO("lwork=%d\n",lwork);
    lwork *= 2;                 // increase lwork to work around ScaLapack possible bug
    REAL_T *work = bml_allocate_memory(lwork * sizeof(REAL_T));;

    //// get lwork value from ScaLapack call
    //// for verification
    //lwork=-1;
    //SYEVD(&jobz, &uplo, &m, atmp, &ione, &ione, desc, ev,
    //      pzmat, &ione, &ione, desc, work, &lwork,
//#if defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
    //      rwork, &lrwork,
//#endif
    //      iwork, &liwork, &info);
    //lwork=work[0];
    //LOG_INFO("lwork=%d\n",lwork);
    //lwork*=2;

    // now solve eigenvalue problem
    SYEVD(&jobz, &uplo, &m, atmp, &ione, &ione, desc, ev,
          pzmat, &ione, &ione, desc, work, &lwork,
#if defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
          rwork, &lrwork,
#endif
          iwork, &liwork, &info);

    if (info > 0)
        LOG_ERROR("Eigenvalue %d did not converge\n", info);
    if (info < 0)
        LOG_ERROR("%d -th argument of SYEVD call had an illegal value\n",
                  info);

    for (int i = 0; i < A->N; i++)
        typed_eigenvalues[i] = (REAL_T) ev[i];

    // clean up
    free(ev);
    bml_free_memory(work);
    bml_free_memory(iwork);
#if defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
    bml_free_memory(rwork);
#endif

    bml_deallocate(&amat);
    if (bml_get_type(Alocal) != dense)
    {
        bml_deallocate(&(eigenvectors->matrix));
        eigenvectors->matrix =
            bml_convert(zmat, bml_get_type(Alocal), A->matrix_precision,
                        A->M / A->npcols, sequential);
        bml_deallocate(&zmat);
    }
    // transpose eigenvectors to have them stored row-major
    bml_transpose(eigenvectors->matrix);
#else
    LOG_ERROR
        ("Build with ScaLAPACK required for distributed2d diagonalization\n");
#endif
    return;
}
