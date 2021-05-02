#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_parallel_csr.h"
#include "bml_types_csr.h"

#include <stdlib.h>
#include <string.h>

#ifdef DO_MPI
void
bml_mpi_send_csr(
    bml_matrix_csr_t * A,
    const int dst,
    MPI_Comm comm)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_send_csr_single_real(A, dst, comm);
            break;
        case double_real:
            bml_mpi_send_csr_double_real(A, dst, comm);
            break;
        case single_complex:
            bml_mpi_send_csr_single_complex(A, dst, comm);
            break;
        case double_complex:
            bml_mpi_send_csr_double_complex(A, dst, comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_mpi_recv_csr(
    bml_matrix_csr_t * A,
    const int dst,
    MPI_Comm comm)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_recv_csr_single_real(A, dst, comm);
            break;
        case double_real:
            bml_mpi_recv_csr_double_real(A, dst, comm);
            break;
        case single_complex:
            bml_mpi_recv_csr_single_complex(A, dst, comm);
            break;
        case double_complex:
            bml_mpi_recv_csr_double_complex(A, dst, comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

bml_matrix_csr_t *
bml_mpi_recv_matrix_csr(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_mpi_recv_matrix_csr_single_real(N, M, src, comm);
            break;
        case double_real:
            return bml_mpi_recv_matrix_csr_double_real(N, M, src, comm);
            break;
        case single_complex:
            return bml_mpi_recv_matrix_csr_single_complex(N, M, src, comm);
            break;
        case double_complex:
            return bml_mpi_recv_matrix_csr_double_complex(N, M, src, comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

#endif
