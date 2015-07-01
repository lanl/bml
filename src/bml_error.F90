!> @copyright Los Alamos National Laboratory 2015

!> A module for error handling in bml.
module bml_error

  implicit none

contains

  !> Common error handling of bml. This function writes out an error
  !! message and exits.
  !!
  !! In the future one could imagine something more like exceptions,
  !! in which the error gets passed up the call stack.
  !!
  !! @param file The filename in which the error occurred.
  !! @param line The line number in that file.
  !! @param message The error message.
  subroutine error(file, line, message)

    character(len=*), intent(in) :: file, message
    integer, intent(in) :: line

    character(len=1000) :: line_string

    write(line_string, *) line
    write(*, "(A)") "["//trim(file)//":"//trim(adjustl(line_string))//" FATAL] "//trim(message)

    error stop

  end subroutine error

  !> Common error handling of bml. This function writes out a
  !! non-fatal warning message.
  !!
  !! In the future one could imagine something more like exceptions,
  !! in which the error gets passed up the call stack.
  !!
  !! @param file The filename in which the error occurred.
  !! @param line The line number in that file.
  !! @param message The error message.
  subroutine warning(file, line, message)

    character(len=*), intent(in) :: file, message
    integer, intent(in) :: line

    character(len=1000) :: line_string

    write(line_string, *) line
    write(*, "(A)") "["//trim(file)//":"//trim(adjustl(line_string))//" WARNING] "//trim(message)

  end subroutine warning

end module bml_error
