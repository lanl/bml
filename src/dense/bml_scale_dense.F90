!> \copyright Los Alamos National Laboratory 2015

!> Matrix scaling for dense matrices.
module bml_scale_dense
  implicit none
contains

  !> Scale a dense bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param A The matrix
  !! \param C The matrix
  subroutine scale_dense(alpha, A, C)

    use bml_type_dense

    double precision, intent(in) :: alpha
    type(bml_matrix_dense_t), intent(in) :: A
    type(bml_matrix_dense_t), intent(out) :: C

    C%matrix = alpha*A%matrix

  end subroutine scale_dense

end module bml_scale_dense
