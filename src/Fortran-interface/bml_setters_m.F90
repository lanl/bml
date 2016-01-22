module bml_setters_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  interface bml_set_row
     module procedure bml_set_row_single_real
     module procedure bml_set_row_double_real
     module procedure bml_set_row_single_complex
     module procedure bml_set_row_double_complex
  end interface bml_set_row

  interface bml_set_diag
     module procedure bml_set_diag_single_real
     module procedure bml_set_diag_double_real
     module procedure bml_set_diag_single_complex
     module procedure bml_set_diag_double_complex
  end interface bml_set_diag

  public :: bml_set_row
  public :: bml_set_diag

contains

  subroutine bml_set_row_single_real(a, i, row)

    type(bml_matrix_t), intent(inout) :: a
    integer(C_INT), intent(in) :: i
    real(C_FLOAT), target, intent(in) :: row(*)

    call bml_set_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_set_row_single_real

  subroutine bml_set_row_double_real(a, i, row)

    type(bml_matrix_t), intent(inout) :: a
    integer(C_INT), intent(in) :: i
    real(C_DOUBLE), target, intent(in) :: row(*)

    call bml_set_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_set_row_double_real

  subroutine bml_set_row_single_complex(a, i, row)

    type(bml_matrix_t), intent(inout) :: a
    integer(C_INT), intent(in) :: i
    complex(C_FLOAT_COMPLEX), target, intent(in) :: row(*)

    call bml_set_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_set_row_single_complex

  subroutine bml_set_row_double_complex(a, i, row)

    type(bml_matrix_t), intent(inout) :: a
    integer(C_INT), intent(in) :: i
    complex(C_DOUBLE_COMPLEX), target, intent(in) :: row(*)

    call bml_set_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_set_row_double_complex

end module bml_setters_m
