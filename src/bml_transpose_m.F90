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
    use bml_type_dense_m
    use bml_transpose_dense_m

    class(bml_matrix_t), pointer, intent(in) :: a
    class(bml_matrix_t), pointer :: a_t

    if(.not. associated(a)) then
       call bml_error(__FILE__, __LINE__, "matrix is not associated")
    end if

    select type(a)
    class is(bml_matrix_dense_t)
       a_t => bml_transpose_dense(a)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end function bml_transpose

end module bml_transpose_m
