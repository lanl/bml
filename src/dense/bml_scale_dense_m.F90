!> \copyright Los Alamos National Laboratory 2015

!> Matrix scaling for dense matrices.
module bml_scale_dense_m
  implicit none
contains

  !> Scale a dense bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param A The matrix
  subroutine scale_one_dense(alpha, A)

    use bml_type_dense_m

    double precision, intent(in) :: alpha
    type(bml_matrix_dense_t), intent(inout) :: A

    A%matrix = alpha*A%matrix

  end subroutine scale_one_dense

  !> Scale a dense bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param A The matrix
  !! \param C The matrix
  subroutine scale_two_dense(alpha, A, C)

    use bml_type_dense_m

    double precision, intent(in) :: alpha
    type(bml_matrix_dense_t), intent(in) :: A
    type(bml_matrix_dense_t), intent(inout) :: C

    C%matrix = alpha*A%matrix

  end subroutine scale_two_dense

end module bml_scale_dense_m
