!> \copyright Los Alamos National Laboratory 2015

!> Matrix multiplication for dense matrices.
module bml_multiply_dense_m
  implicit none
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

    C%matrix = alpha*matmul(A%matrix, B%matrix)+beta*C%matrix

  end subroutine multiply_dense

end module bml_multiply_dense_m
