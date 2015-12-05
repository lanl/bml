module bml_fc_tools_m

  use bml_c_interface_m, only : C_NULL_CHAR, C_CHAR

  implicit none
  private

  public :: f_c_string

  integer, parameter :: LEN_C_NULL_CHAR = len(C_NULL_CHAR)

contains

  !> Returns a NULL terminated C-style string.
  !!
  !! \param fstr  Fortran string.
  !! \return cstr  The Fortran string terminated with C NULL.
  pure function f_c_string(fstr) result(cstr)

    character(len=*, kind=C_CHAR), intent(in) :: fstr
    character(len=len_f_c_string(fstr), kind=C_CHAR) :: cstr

    cstr = trim(fstr)//C_NULL_CHAR

  end function f_c_string

  !> Returns the trimmed length of the string with NULL termination added.
  pure function len_f_c_string(fstr) result(strlen)

    character(len=*, kind=C_CHAR), intent(in) :: fstr
    integer :: strlen

    strlen = len_trim(fstr) + LEN_C_NULL_CHAR

  end function len_f_c_string

end module bml_fc_tools_m
