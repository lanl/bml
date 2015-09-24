!> Matrix scaling for matrices.
module bml_scale_m

  implicit none

  private

  !> Scale a matrix.
  interface bml_scale
     module procedure scale_one
     module procedure scale_two
  end interface bml_scale

  public :: bml_scale

contains

  !> Scale a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param a The matrix
  subroutine scale_one(alpha, a)

    use bml_types

    double precision, intent(in) :: alpha
    class(bml_matrix_t), intent(inout) :: a

  end subroutine scale_one

  !> Scale a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A \f$
  !!
  !! \param alpha The factor
  !! \param a The matrix
  !! \param c The matrix
  subroutine scale_two(alpha, a, c)

    use bml_types

    double precision, intent(in) :: alpha
    class(bml_matrix_t), intent(in) :: a
    class(bml_matrix_t), allocatable, intent(out) :: c

  end subroutine scale_two

end module bml_scale_m
