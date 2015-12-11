module bml_getters_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  interface bml_get_row
     module procedure bml_get_row_single_real
     module procedure bml_get_row_double_real
     module procedure bml_get_row_single_complex
     module procedure bml_get_row_double_complex
  end interface bml_get_row

  public :: bml_get_row

contains

  subroutine bml_get_row_single_real(a, i, row)

    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    real(C_FLOAT), target, intent(out) :: row(*)

    call bml_get_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_get_row_single_real


  subroutine bml_get_row_double_real(a, i, row)

    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    real(C_DOUBLE), target, intent(out) :: row(*)

    call bml_get_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_get_row_double_real


  subroutine bml_get_row_single_complex(a, i, row)

    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    complex(C_FLOAT_COMPLEX), target, intent(out) :: row(*)

    call bml_get_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_get_row_single_complex

  subroutine bml_get_row_double_complex(a, i, row)

    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    complex(C_DOUBLE_COMPLEX), target, intent(out) :: row(*)

    call bml_get_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_get_row_double_complex

end module bml_getters_m
