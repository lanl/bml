!> \copyright Los Alamos National Laboratory 2015

!> Matrix multiplication for dense matrices.
module bml_multiply_dense_m

  implicit none

  private

  !> Interface to BLAS {s,d}gemm functions.
  interface
#ifdef HAVE_DGEMM
     subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
       double precision :: alpha, beta
       integer :: k, lda, ldb, ldc, m, n
       character :: transa, transb
       double precision :: a(lda,*), b(ldb,*), c(ldc,*)
     end subroutine dgemm
#endif
#ifdef HAVE_SGEMM
     subroutine sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
       real :: alpha, beta
       integer :: k, lda, ldb, ldc, m, n
       character :: transa, transb
       real :: a(lda,*), b(ldb,*), c(ldc,*)
     end subroutine sgemm
#endif
  end interface

  interface bml_multiply_dense
     module procedure multiply_dense_single
     module procedure multiply_dense_double
  end interface bml_multiply_dense

  public :: bml_multiply_dense

contains

  !> Multiply two matrices.
  !!
  !! \f$ C \leftarrow \alpha A \times B + \beta C \f$
  !!
  !! \param a Matrix \f$ A \f$.
  !! \param b Matrix \f$ B \f$.
  !! \param c Matrix \f$ C \f$.
  !! \param alpha The factor \f$ \alpha \f$.
  !! \param beta The factor \f$ \beta \f$.
  subroutine multiply_dense_single(a, b, c, alpha, beta)

    use bml_type_dense_m

    type(bml_matrix_dense_single_t), intent(in) :: a, b
    type(bml_matrix_dense_single_t), intent(inout) :: c
    double precision, intent(in) :: alpha
    double precision, intent(in) :: beta

    real :: alpha_
    real :: beta_

    alpha_ = alpha
    beta_ = beta

#ifdef HAVE_SGEMM
    call sgemm("N", "N", a%n, a%n, a%n, alpha_, a%matrix, a%n, b%matrix, a%n, beta_, c%matrix, a%n)
#else
    c%matrix = alpha*matmul(a%matrix, b%matrix)+beta*c%matrix
#endif

  end subroutine multiply_dense_single

  !> Multiply two matrices.
  !!
  !! \f$ C \leftarrow \alpha A \times B + \beta C \f$
  !!
  !! \param a Matrix \f$ A \f$.
  !! \param b Matrix \f$ B \f$.
  !! \param c Matrix \f$ C \f$.
  !! \param alpha The factor \f$ \alpha \f$.
  !! \param beta The factor \f$ \beta \f$.
  subroutine multiply_dense_double(a, b, c, alpha, beta)

    use bml_type_dense_m

    type(bml_matrix_dense_double_t), intent(in) :: a, b
    type(bml_matrix_dense_double_t), intent(inout) :: c
    double precision, intent(in) :: alpha
    double precision, intent(in) :: beta

#ifdef HAVE_DGEMM
    call dgemm("N", "N", a%n, a%n, a%n, alpha, a%matrix, a%n, b%matrix, a%n, beta, c%matrix, a%n)
#else
    c%matrix = alpha*matmul(a%matrix, b%matrix)+beta*c%matrix
#endif

  end subroutine multiply_dense_double

end module bml_multiply_dense_m
