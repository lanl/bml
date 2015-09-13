  !> Utility matrix functions.
module bml_utilities

  implicit none

  private

  interface

     subroutine bml_print_matrix_single_C(n, a, i_l, i_u, j_l, j_u, matrix_precision) &
          bind(C, name="bml_print_matrix_single")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: n
       real(C_FLOAT), intent(in) :: a(:, :)
       integer(C_INT), value, intent(in) :: i_l
       integer(C_INT), value, intent(in) :: i_u
       integer(C_INT), value, intent(in) :: j_l
       integer(C_INT), value, intent(in) :: j_u
     end subroutine bml_print_matrix_single_C

     subroutine bml_print_matrix_double_C(n, a, i_l, i_u, j_l, j_u, matrix_precision) &
          bind(C, name="bml_print_matrix_double")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: n
       real(C_DOUBLE), intent(in) :: a(:, :)
       integer(C_INT), value, intent(in) :: i_l
       integer(C_INT), value, intent(in) :: i_u
       integer(C_INT), value, intent(in) :: j_l
       integer(C_INT), value, intent(in) :: j_u
     end subroutine bml_print_matrix_double_C

  end interface

  interface bml_print_matrix
     module procedure bml_print_matrix_single
     module procedure bml_print_matrix_double
  end interface bml_print_matrix

  public :: bml_print_matrix

contains

  subroutine bml_print_matrix_single(tag, a, i_l, i_u, j_l, j_u)

    character(len=*), intent(in) :: tag
    real, intent(in) :: a(:, :)
    integer, optional, intent(in) :: i_l
    integer, optional, intent(in) :: i_u
    integer, optional, intent(in) :: j_l
    integer, optional, intent(in) :: j_u

  end subroutine bml_print_matrix_single

  subroutine bml_print_matrix_double(tag, a, i_l, i_u, j_l, j_u)

    character(len=*), intent(in) :: tag
    double precision, intent(in) :: a(:, :)
    integer, optional, intent(in) :: i_l
    integer, optional, intent(in) :: i_u
    integer, optional, intent(in) :: j_l
    integer, optional, intent(in) :: j_u

  end subroutine bml_print_matrix_double

end module bml_utilities
