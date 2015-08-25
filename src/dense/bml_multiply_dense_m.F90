!> \copyright Los Alamos National Laboratory 2015

!> Matrix multiplication for dense matrices.
module bml_multiply_dense_m
  implicit none

  !> Interface to BLAS {s,d}gemm functions.
  interface
     subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
       double precision :: alpha, beta
       integer :: k, lda, ldb, ldc, m, n
       character :: transa, transb
       double precision :: a(lda,*), b(ldb,*), c(ldc,*)
     end subroutine dgemm
     subroutine sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
       real :: alpha, beta
       integer :: k, lda, ldb, ldc, m, n
       character :: transa, transb
       real :: a(lda,*), b(ldb,*), c(ldc,*)
     end subroutine sgemm
  end interface

contains

  !> Multiply two matrices.
  !!
  !! \f$ C \leftarrow \alpha A \times B + \beta C \f$
  !!
  !! \param A Matrix \f$ A \f$.
  !! \param B Matrix \f$ B \f$.
  !! \param C Matrix \f$ C \f$.
  !! \param alpha The factor \f$ \alpha \f$.
  !! \param beta The factor \f$ \beta \f$.
  subroutine multiply_dense(A, B, C, alpha, beta)

    use bml_type_dense_m

    type(bml_matrix_dense_double_t), intent(in) :: A, B
    type(bml_matrix_dense_double_t), intent(inout) :: C
    double precision, intent(in) :: alpha
    double precision, intent(in) :: beta

#ifdef BLAS_FOUND
    call dgemm("N", "N", A%N, A%N, A%N, alpha, A%matrix, A%N, B%matrix, A%N, beta, C%matrix, A%N)
#else
    C%matrix = alpha*matmul(A%matrix, B%matrix)+beta*C%matrix
#endif

  end subroutine multiply_dense

end module bml_multiply_dense_m
