  !> Utility matrix functions.
module bml_utilities

  implicit none

  private

  interface

     subroutine bml_print_matrix_C(n, matrix_precision, a, i_l, i_u, j_l, j_u) &
          bind(C, name="bml_print_matrix")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: matrix_precision
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i_l
       integer(C_INT), value, intent(in) :: i_u
       integer(C_INT), value, intent(in) :: j_l
       integer(C_INT), value, intent(in) :: j_u
     end subroutine bml_print_matrix_C

  end interface

  interface bml_print_matrix
     module procedure bml_print_matrix_single
     module procedure bml_print_matrix_double
  end interface bml_print_matrix

  public :: bml_print_matrix

contains

  subroutine bml_print_matrix_single(tag, a, i_l, i_u, j_l, j_u)

    use, intrinsic :: iso_C_binding
    use bml_types
    use bml_interface

    character(len=*), intent(in) :: tag
    real, target, intent(in) :: a(:, :)
    integer, intent(in) :: i_l
    integer, intent(in) :: i_u
    integer, intent(in) :: j_l
    integer, intent(in) :: j_u

    associate(a_ptr => a(lbound(a, 1), lbound(a, 2)))
      call bml_print_matrix_C(size(a, 1), get_enum_id(BML_PRECISION_SINGLE), &
           c_loc(a_ptr), i_l-1, i_u, j_l-1, j_u)
    end associate

  end subroutine bml_print_matrix_single

  subroutine bml_print_matrix_double(tag, a, i_l, i_u, j_l, j_u)

    use, intrinsic :: iso_C_binding
    use bml_types
    use bml_interface

    character(len=*), intent(in) :: tag
    double precision, target, intent(in) :: a(:, :)
    integer, intent(in) :: i_l
    integer, intent(in) :: i_u
    integer, intent(in) :: j_l
    integer, intent(in) :: j_u

    associate(a_ptr => a(lbound(a, 1), lbound(a, 2)))
      call bml_print_matrix_C(size(a, 1), get_enum_id(BML_PRECISION_DOUBLE), &
           c_loc(a_ptr), i_l-1, i_u, j_l-1, j_u)
    end associate

  end subroutine bml_print_matrix_double

end module bml_utilities
