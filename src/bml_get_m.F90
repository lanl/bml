!> \copyright Los Alamos National Laboratory 2015

!> Getter functions for matrix elements.
module bml_get_m

  implicit none

  private

  public :: bml_get

contains

  !> Get a matrix element.
  !!
  !! @param a The matrix.
  !! @param i The row index.
  !! @param j The column index.
  !! @return The matrix element.
  function bml_get(a, i, j) result(a_ij)

    use bml_type_m

    class(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i, j
    double precision :: a_ij

  end function bml_get

end module bml_get_m
