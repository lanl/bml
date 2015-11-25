module bml_fc_tools_m
  use, intrinsic :: iso_c_binding
  implicit none
  private

  public :: f_c_string

  integer, parameter :: LEN_C_NULL_CHAR = len(c_null_char)

contains

  !> Returns a NULL terminated C-style string.
  !!
  !! \param fstr  Fortran string.
  !! \return cstr  The Fortran string terminated with C NULL.
  !!
  pure function f_c_string(fstr) result(cstr)
    character(len=*, kind=c_char), intent(in) :: fstr
    character(len=len_f_c_string(fstr), kind=c_char) :: cstr

    cstr = trim(fstr) // c_null_char

  end function f_c_string

  
  !! Returns the trimmed length of the string with NULL termination added.
  pure function len_f_c_string(fstr) result(strlen)
    character(len=*, kind=c_char), intent(in) :: fstr
    integer :: strlen

    strlen = len_trim(fstr) + LEN_C_NULL_CHAR

  end function len_f_c_string


end module bml_fc_tools_m
