!> Utility matrix functions.
module bml_utilities_MATRIX_TYPE_m

  implicit none

  private

  interface
     subroutine bml_print_dense_matrix_C(n, matrix_precision, a, i_l, i_u, j_l, j_u) &
          bind(C, name="bml_print_dense_matrix")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: matrix_precision
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i_l
       integer(C_INT), value, intent(in) :: i_u
       integer(C_INT), value, intent(in) :: j_l
       integer(C_INT), value, intent(in) :: j_u
     end subroutine bml_print_dense_matrix_C
  end interface

  !> Print a dense matrix.
  interface bml_print_matrix
     module procedure bml_print_dense_matrix_MATRIX_TYPE
  end interface bml_print_matrix

  public :: bml_print_matrix

contains

  !> Print a dense matrix.
  !!
  !! \param tag A string to print before the matrix.
  !! \param a The matrix.
  !! \param i_l The lower row bound.
  !! \param i_u The upper row bound.
  !! \param j_l The lower column bound.
  !! \param j_u The upper column bound.
  subroutine bml_print_dense_matrix_MATRIX_TYPE(tag, a, i_l, i_u, j_l, j_u)

    use, intrinsic :: iso_C_binding
    use bml_types_m
    use bml_interface_m

    character(len=*), intent(in) :: tag
    REAL_TYPE, target, intent(in) :: a(:, :)
    integer, intent(in) :: i_l
    integer, intent(in) :: i_u
    integer, intent(in) :: j_l
    integer, intent(in) :: j_u

    associate(a_ptr => a(lbound(a, 1), lbound(a, 2)))
      ! Print bounds are inclusive here, i.e. [i_l, i_u], but are
      ! exclusive in the upper bound in the C code.
      call bml_print_dense_matrix_C(size(a, 1), get_enum_id(PRECISION_NAME), &
           c_loc(a_ptr), &
           i_l-lbound(a, 1), i_u-lbound(a, 1)+1, &
           j_l-lbound(a, 2), j_u-lbound(a, 2)+1)
    end associate

  end subroutine bml_print_dense_matrix_MATRIX_TYPE

end module bml_utilities_MATRIX_TYPE_m
