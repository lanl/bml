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
  !! \param a The matrix
  subroutine scale_one_dense(alpha, a)

    use bml_type_dense_m

    double precision, intent(in) :: alpha
    type(bml_matrix_dense_double_t), intent(inout) :: a

    a%matrix = alpha*a%matrix

  end subroutine scale_one_dense

  !> Scale a dense bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param a The matrix
  !! \param c The matrix
  subroutine scale_two_dense(alpha, a, c)

    use bml_type_dense_m

    double precision, intent(in) :: alpha
    type(bml_matrix_dense_double_t), intent(in) :: a
    type(bml_matrix_dense_double_t), intent(inout) :: c

    c%matrix = alpha*a%matrix

  end subroutine scale_two_dense

end module bml_scale_dense_m
