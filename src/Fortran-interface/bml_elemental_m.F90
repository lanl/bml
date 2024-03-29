module bml_elemental_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  interface bml_get_element
    module procedure bml_get_element_single_real
    module procedure bml_get_element_double_real
#ifdef BML_COMPLEX
    module procedure bml_get_element_single_complex
    module procedure bml_get_element_double_complex
#endif
  end interface bml_get_element

  public :: bml_get_element

contains

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_element_single_real(a_ij, a, i, j)

    real(C_FLOAT), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer(C_INT), intent(in) :: j

    a_ij = bml_get_element_single_real_C(a%ptr, i-1, j-1)

  end subroutine bml_get_element_single_real

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_element_double_real(a_ij, a, i, j)

    real(C_DOUBLE), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer(C_INT), intent(in) :: j

    a_ij = bml_get_element_double_real_C(a%ptr, i-1, j-1)

  end subroutine bml_get_element_double_real

#ifdef BML_COMPLEX
  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_element_single_complex(a_ij, a, i, j)

    complex(C_FLOAT_COMPLEX), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer(C_INT), intent(in) :: j

    a_ij = bml_get_element_single_complex_C(a%ptr, i-1, j-1)

  end subroutine bml_get_element_single_complex

  !> Get a single matrix element.
  !!
  !! \param a_ij The matrix element
  !! \param a The matrix
  !! \param i The row index
  !! \param j The column index
  subroutine bml_get_element_double_complex(a_ij, a, i, j)

    complex(C_DOUBLE_COMPLEX), intent(out) :: a_ij
    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer(C_INT), intent(in) :: j

    a_ij = bml_get_element_double_complex_C(a%ptr, i-1, j-1)

  end subroutine bml_get_element_double_complex
#endif

end module bml_elemental_m
