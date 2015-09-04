!> \copyright Los Alamos National Laboratory 2015

!> Tranpose functions.
module bml_transpose_m
  implicit none
contains

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @return The transpose.
  function bml_transpose(a) result(a_t)

    use bml_type_m

    class(bml_matrix_t), pointer, intent(in) :: a
    class(bml_matrix_t), pointer :: a_t

  end function bml_transpose

end module bml_transpose_m
