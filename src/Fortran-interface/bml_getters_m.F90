module bml_getters_m

  implicit none
  private

  interface

     subroutine bml_get_row_C(a, i, row) bind(C, name="bml_get_row")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       type(C_PTR), value, intent(in) :: row
     end subroutine bml_get_row_C

  end interface

  interface bml_get_row
     module procedure bml_get_row_single_real
     module procedure bml_get_row_double_real
     module procedure bml_get_row_single_complex
     module procedure bml_get_row_double_complex
  end interface bml_get_row

  public :: bml_get_row

contains

  subroutine bml_get_row_single_real(a, i, row)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    real(kind(0e0)), target, intent(out) :: row(*)

    call bml_get_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_get_row_single_real

  subroutine bml_get_row_double_real(a, i, row)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    real(kind(0d0)), target, intent(out) :: row(*)

    call bml_get_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_get_row_double_real

  subroutine bml_get_row_single_complex(a, i, row)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    complex(kind(0e0)), target, intent(out) :: row(*)

    call bml_get_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_get_row_single_complex

  subroutine bml_get_row_double_complex(a, i, row)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    complex(kind(0d0)), target, intent(out) :: row(*)

    call bml_get_row_C(a%ptr, i-1, c_loc(row))

  end subroutine bml_get_row_double_complex

end module bml_getters_m
