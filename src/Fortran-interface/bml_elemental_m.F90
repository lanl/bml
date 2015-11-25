module bml_elemental_m

  implicit none
  private

  interface

     function bml_get_single_real_C(a, i, j) bind(C, name="bml_get_single_real")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       integer(C_INT), value, intent(in) :: j
       real(C_FLOAT) :: bml_get_single_real_C
     end function bml_get_single_real_C

     function bml_get_double_real_C(a, i, j) bind(C, name="bml_get_double_real")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       integer(C_INT), value, intent(in) :: j
       real(C_DOUBLE) :: bml_get_double_real_C
     end function bml_get_double_real_C

     function bml_get_single_complex_C(a, i, j) bind(C, name="bml_get_single_complex")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       integer(C_INT), value, intent(in) :: j
       complex(C_FLOAT_COMPLEX) :: bml_get_single_complex_C
     end function bml_get_single_complex_C

     function bml_get_double_complex_C(a, i, j) bind(C, name="bml_get_double_complex")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       integer(C_INT), value, intent(in) :: j
       complex(C_DOUBLE_COMPLEX) :: bml_get_double_complex_C
     end function bml_get_double_complex_C

  end interface

  interface bml_get
     module procedure bml_get_single_real
     module procedure bml_get_double_real
     module procedure bml_get_single_complex
     module procedure bml_get_double_complex
  end interface bml_get

  public :: bml_get

contains

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_single_real(a_ij, a, i, j)

    use bml_types_m

    real, intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    integer, intent(in) :: j

    a_ij = bml_get_single_real_C(a%ptr, i-1, j-1)

  end subroutine bml_get_single_real

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_double_real(a_ij, a, i, j)

    use bml_types_m

    double precision, intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    integer, intent(in) :: j

    a_ij = bml_get_double_real_C(a%ptr, i-1, j-1)

  end subroutine bml_get_double_real

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_single_complex(a_ij, a, i, j)

    use bml_types_m

    complex(kind(0.0)), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    integer, intent(in) :: j

    a_ij = bml_get_single_complex_C(a%ptr, i-1, j-1)

  end subroutine bml_get_single_complex

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_double_complex(a_ij, a, i, j)

    use bml_types_m

    complex(kind(0d0)), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    integer, intent(in) :: j

    a_ij = bml_get_double_complex_C(a%ptr, i-1, j-1)

  end subroutine bml_get_double_complex

end module bml_elemental_m
