!> \copyright Los Alamos National Laboratory 2015

!> Matrix transpose functions.
module bml_transpose_dense_m
  implicit none
contains

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @return The transpose.
  function bml_transpose_dense(a) result(a_t)

    use bml_type_m
    use bml_type_dense_m

    class(bml_matrix_dense_t), pointer, intent(in) :: a
    class(bml_matrix_dense_t), pointer :: a_t

    select type(a)
    type is(bml_matrix_dense_single_t)
       call allocate_matrix_dense(a%n, a_t, BML_PRECISION_SINGLE)
       select type(a_t)
       type is(bml_matrix_dense_single_t)
          a_t%matrix = transpose(a%matrix)
       end select
    type is(bml_matrix_dense_double_t)
       call allocate_matrix_dense(a%n, a_t, BML_PRECISION_DOUBLE)
       select type(a_t)
       type is(bml_matrix_dense_double_t)
          a_t%matrix = transpose(a%matrix)
       end select
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end function bml_transpose_dense

end module bml_transpose_dense_m
