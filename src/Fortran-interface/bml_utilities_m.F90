!> Utility matrix functions.
module bml_utilities_m

  use bml_utilities_single_real_m
  use bml_utilities_double_real_m
  use bml_utilities_single_complex_m
  use bml_utilities_double_complex_m

  implicit none

  private

  interface

     subroutine bml_print_bml_vector_C(v, i_l, i_u) &
          bind(C, name="bml_print_bml_vector")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: v
       integer(C_INT), value, intent(in) :: i_l
       integer(C_INT), value, intent(in) :: i_u
     end subroutine bml_print_bml_vector_C

     subroutine bml_print_bml_matrix_C(a, i_l, i_u, j_l, j_u) &
          bind(C, name="bml_print_bml_matrix")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i_l
       integer(C_INT), value, intent(in) :: i_u
       integer(C_INT), value, intent(in) :: j_l
       integer(C_INT), value, intent(in) :: j_u
     end subroutine bml_print_bml_matrix_C

     subroutine bml_read_bml_matrix_C(a, filename) &
          bind(C, name="bml_read_bml_matrix")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       character(C_CHAR), intent(in) :: filename(*)
     end subroutine bml_read_bml_matrix_C

     subroutine bml_write_bml_matrix_C(a, filename) &
          bind(C, name="bml_write_bml_matrix")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       character(C_CHAR), intent(in) :: filename(*)
     end subroutine bml_write_bml_matrix_C

  end interface

  interface bml_print_vector
     module procedure bml_print_bml_vector
  end interface bml_print_vector

  interface bml_print_matrix
     module procedure bml_print_bml_matrix
  end interface bml_print_matrix

  interface bml_read_matrix
     module procedure bml_read_bml_matrix
  end interface bml_read_matrix

  interface bml_write_matrix
     module procedure bml_write_bml_matrix
  end interface bml_write_matrix

  public :: bml_print_vector
  public :: bml_print_matrix
  public :: bml_read_matrix
  public :: bml_write_matrix

contains

  !> Print a bml matrix.
  !!
  !! \param tag A string to print before the matrix.
  !! \param a The matrix.
  !! \param i_l The lower row bound.
  !! \param i_u The upper row bound.
  !! \param j_l The lower column bound.
  !! \param j_u The upper column bound.
  subroutine bml_print_bml_matrix(tag, a, i_l, i_u, j_l, j_u)

    use, intrinsic :: iso_C_binding
    use bml_types_m

    character(len=*), intent(in) :: tag
    type(bml_matrix_t), target, intent(in) :: a
    integer, intent(in) :: i_l
    integer, intent(in) :: i_u
    integer, intent(in) :: j_l
    integer, intent(in) :: j_u

    write(*, "(A)") tag
    call bml_print_bml_matrix_C(a%ptr, i_l, i_u, j_l, j_u)

  end subroutine bml_print_bml_matrix

  !> Print a bml vector.
  !!
  !! \param tag A string to print before the matrix.
  !! \param v The vector.
  !! \param i_l The lower row bound.
  !! \param i_u The upper row bound.
  subroutine bml_print_bml_vector(tag, v, i_l, i_u)

    use, intrinsic :: iso_C_binding
    use bml_types_m

    character(len=*), intent(in) :: tag
    type(bml_vector_t), target, intent(in) :: v
    integer, intent(in) :: i_l
    integer, intent(in) :: i_u

    call bml_print_bml_vector_C(v%ptr, i_l, i_u)

  end subroutine bml_print_bml_vector

  !> Read in a bml matrix from a Matrix Market file.
  !!
  !! \param a The matrix.
  !! \param filename The file to read from.
  subroutine bml_read_bml_matrix(a, filename)

    use, intrinsic :: iso_C_binding
    use bml_types_m

    character(len=*), intent(in) :: filename
    type(bml_matrix_t), intent(in) :: a

    call bml_read_bml_matrix_C(a%ptr, filename)

  end subroutine bml_read_bml_matrix

  !> Write out a bml matrix to a Matrix Market file.
  !!
  !! \param a The matrix.
  !! \param filename The file to write to.
  subroutine bml_write_bml_matrix(a, filename)

    use, intrinsic :: iso_C_binding
    use bml_types_m

    character(len=*), intent(in) :: filename
    type(bml_matrix_t), intent(in) :: a

    call bml_write_bml_matrix_C(a%ptr, filename)

  end subroutine bml_write_bml_matrix

end module bml_utilities_m
