module bml_elemental_m
  use, intrinsic :: iso_c_binding
  use bml_types_m
  implicit none
  private

  interface

     function bml_get_single_real_C(a, i, j) bind(C, name="bml_get_single_real")
       import :: C_PTR, C_INT, C_FLOAT
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       integer(C_INT), value, intent(in) :: j
       real(C_FLOAT) :: bml_get_single_real_C
     end function bml_get_single_real_C

     function bml_get_double_real_C(a, i, j) bind(C, name="bml_get_double_real")
       import :: C_PTR, C_INT, C_DOUBLE
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       integer(C_INT), value, intent(in) :: j
       real(C_DOUBLE) :: bml_get_double_real_C
     end function bml_get_double_real_C

     function bml_get_single_complex_C(a, i, j) bind(C, name="bml_get_single_complex")
       import :: C_PTR, C_INT, C_FLOAT_COMPLEX
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       integer(C_INT), value, intent(in) :: j
       complex(C_FLOAT_COMPLEX) :: bml_get_single_complex_C
     end function bml_get_single_complex_C

     function bml_get_double_complex_C(a, i, j) bind(C, name="bml_get_double_complex")
       import :: C_PTR, C_INT, C_DOUBLE_COMPLEX
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

    real(C_FLOAT), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer(C_INT), intent(in) :: j

    a_ij = bml_get_single_real_C(a%ptr, i-1, j-1)

  end subroutine bml_get_single_real

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_double_real(a_ij, a, i, j)

    real(C_DOUBLE), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer(C_INT), intent(in) :: j

    a_ij = bml_get_double_real_C(a%ptr, i-1, j-1)

  end subroutine bml_get_double_real

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_single_complex(a_ij, a, i, j)

    complex(C_FLOAT_COMPLEX), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer(C_INT), intent(in) :: j

    a_ij = bml_get_single_complex_C(a%ptr, i-1, j-1)

  end subroutine bml_get_single_complex

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_double_complex(a_ij, a, i, j)

    complex(C_DOUBLE_COMPLEX), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer(C_INT), intent(in) :: j

    a_ij = bml_get_double_complex_C(a%ptr, i-1, j-1)

  end subroutine bml_get_double_complex

end module bml_elemental_m
