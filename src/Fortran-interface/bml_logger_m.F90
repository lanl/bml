module bml_logger_m

  use bml_c_interface_m

  implicit none
  private
  public :: bml_print_version

contains

  subroutine bml_print_version()
    call bml_print_version_C()
  end subroutine bml_print_version

end module bml_logger_m
