#ifdef BML_USE_ELPA
#include <elpa/elpa.h>
#include <elpa/elpa_generic.h>
// ELPA define a macro double_complex that conflicts with ours.
// So we undef it here since we don't use it when interfacing with ELPA
#undef double_complex
#endif

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
#include "../bml_transpose.h"
#include "../bml_copy.h"

#ifdef BML_USE_ELPA
#include "../dense/bml_allocate_dense.h"
#ifdef BML_USE_MAGMA
#include "magma_v2.h"
#endif
#endif

#include <mpi.h>

#include <complex.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "../../typed.h"

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
#ifdef BML_USE_SCALAPACK
void TYPED_FUNC(
    bml_diagonalize_distributed2d_scalapack) (
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors)
{
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
    return;
}
#endif

#ifdef BML_USE_ELPA
// Yu, V.; Moussa, J.; Kus, P.; Marek, A.; Messmer, P.; Yoon, M.; Lederer, H.; Blum, V.
// "GPU-Acceleration of the ELPA2 Distributed Eigensolver for Dense Symmetric and Hermitian Eigenproblems",
// Computer Physics Communications, 262, 2021
void TYPED_FUNC(
    bml_diagonalize_distributed2d_elpa) (
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors)
{
    char order = 'R';
    int np_rows = A->nprows;
    int np_cols = A->npcols;
    int my_prow = A->myprow;
    int my_pcol = A->mypcol;
    int my_blacs_ctxt = Csys2blacs_handle(A->comm);
    Cblacs_gridinit(&my_blacs_ctxt, &order, np_rows, np_cols);
    Cblacs_gridinfo(my_blacs_ctxt, &np_rows, &np_cols, &my_prow, &my_pcol);

    int na = A->N;
    int na_rows = na / np_rows;
    int na_cols = na / np_cols;
    if (na_rows * np_rows != na)
    {
        LOG_ERROR("Number of MPI tasks/row should divide matrix size\n");
    }
    //printf("Matrix size: %d\n", na);
    //printf("Number of MPI process rows: %d\n", np_rows);
    //printf("Number of MPI process cols: %d\n", np_cols);

    if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
    {
        LOG_ERROR("Error: ELPA API version not supported");
    }

    int error_elpa;
    elpa_t handle = elpa_allocate(&error_elpa);
    /* Set parameters */
    elpa_set(handle, "na", (int) na, &error_elpa);
    assert(error_elpa == ELPA_OK);

    elpa_set(handle, "nev", (int) na, &error_elpa);
    assert(error_elpa == ELPA_OK);

    elpa_set(handle, "local_nrows", (int) na_rows, &error_elpa);
    assert(error_elpa == ELPA_OK);

    elpa_set(handle, "local_ncols", (int) na_cols, &error_elpa);
    assert(error_elpa == ELPA_OK);

    // use one block/MPI task, so sets block size to no. local rows
    elpa_set(handle, "nblk", (int) na_rows, &error_elpa);
    assert(error_elpa == ELPA_OK);

    elpa_set(handle, "mpi_comm_parent", (int) (MPI_Comm_c2f(A->comm)),
             &error_elpa);
    assert(error_elpa == ELPA_OK);

    elpa_set(handle, "process_row", (int) my_prow, &error_elpa);
    assert(error_elpa == ELPA_OK);

    elpa_set(handle, "process_col", (int) my_pcol, &error_elpa);
    assert(error_elpa == ELPA_OK);

    MPI_Barrier(MPI_COMM_WORLD);

    int success = elpa_setup(handle);
    assert(success == ELPA_OK);

    elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error_elpa);
    assert(error_elpa == ELPA_OK);

    elpa_set(handle, "gpu", 1, &error_elpa);
    assert(error_elpa == ELPA_OK);

    bml_matrix_t *Alocal = A->matrix;

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

    // transpose to satisfy column major ELPA convention
    // (global matrix assumed symmetric, so no need for communications)
    if (A->myprow != A->mypcol)
        bml_transpose(amat);

    REAL_T *z = bml_get_data_ptr(zmat);
    assert(z != NULL);
    REAL_T *a = bml_get_data_ptr(amat);
    assert(a != NULL);

    /* Solve EV problem */
    // interface: see elpa_generic.h
    // handle  handle of the ELPA object, which defines the problem
    // a       device pointer to matrix a in GPU memory
    // ev      on return: pointer to eigenvalues in GPU memory
    // q       on return: pointer to eigenvectors in GPU memory
    // error   on return the error code, which can be queried with elpa_strerr()
    LOG_DEBUG("Call ELPA eigensolver");
#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
    float *ev;
    magma_int_t ret = magma_smalloc(&ev, na);
#else
    double *ev;
    magma_int_t ret = magma_dmalloc(&ev, na);
#endif
    assert(ret == MAGMA_SUCCESS);
#if defined(SINGLE_REAL)
    elpa_eigenvectors_float(handle, a, ev, z, &error_elpa);
#endif
#if defined(DOUBLE_REAL)
    elpa_eigenvectors_double(handle, a, ev, z, &error_elpa);
#endif
#if defined(SINGLE_COMPLEX)
    elpa_eigenvectors_float_complex(handle, a, ev, z, &error_elpa);
#endif
#if defined(DOUBLE_COMPLEX)
    elpa_eigenvectors_double_complex(handle, a, ev, z, &error_elpa);
#endif

    assert(error_elpa == ELPA_OK);
    // copy eigenvalues to CPU
    LOG_DEBUG("copy eigenvalues to CPU");
#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
    float *tmp = malloc(na * sizeof(float));
    magma_sgetvector(na, ev, 1, tmp, 1, bml_queue());
#endif
#if defined(DOUBLE_REAL) || defined(DOUBLE_COMPLEX)
    double *tmp = malloc(na * sizeof(double));
    magma_dgetvector(na, ev, 1, tmp, 1, bml_queue());
#endif
    magma_queue_sync(bml_queue());

    REAL_T *ev_ptr = eigenvalues;
    for (int i = 0; i < A->N; i++)
        ev_ptr[i] = (REAL_T) tmp[i];
    free(tmp);

    magma_free(ev);

    bml_deallocate(&amat);
    if (bml_get_type(Alocal) != dense)
    {
        bml_deallocate(&(eigenvectors->matrix));
        eigenvectors->matrix =
            bml_convert(zmat, bml_get_type(Alocal), A->matrix_precision,
                        A->M / A->npcols, sequential);
        bml_deallocate(&zmat);
    }

    elpa_deallocate(handle, &error_elpa);
}
#endif

void TYPED_FUNC(
    bml_diagonalize_distributed2d) (
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors)
{
#ifdef BML_USE_ELPA
    TYPED_FUNC(bml_diagonalize_distributed2d_elpa) (A, eigenvalues,
                                                    eigenvectors);
#else
#ifdef BML_USE_SCALAPACK
    TYPED_FUNC(bml_diagonalize_distributed2d_scalapack) (A, eigenvalues,
                                                         eigenvectors);
#else
    LOG_ERROR
        ("Build with ScaLAPACK required for distributed2d diagonalization\n");
#endif
#endif
    // transpose eigenvectors to have them stored row-major
    bml_transpose(eigenvectors->matrix);
}
