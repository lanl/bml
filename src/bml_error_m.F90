!> \copyright Los Alamos National Laboratory 2015

!> A module for error handling in bml.
module bml_error_m
  implicit none
  private

  public :: error, warning, debug

contains

  !> Common error handling of bml. This function writes out an error
  !! message and exits.
  !!
  !! In the future one could imagine something more like exceptions,
  !! in which the error gets passed up the call stack.
  !!
  !! \param file The filename in which the error occurred.
  !! \param line The line number in that file.
  !! \param tag The message tag.
  !! \param message The error message.
  subroutine bml_msg(file, line, tag, message)

    character(len=*), intent(in) :: file, message, tag
    integer, intent(in) :: line

    character(len=1000) :: line_string

    write(line_string, *) line
    write(*, "(A)") "["//trim(adjustl(file)) &
         //":"//trim(adjustl(line_string)) &
         //" "//trim(adjustl(tag))//"] " &
         //trim(adjustl(message))

  end subroutine bml_msg

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

    call bml_msg(file, line, "ERROR", message)
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

    call bml_msg(file, line, "WARNING", message)

  end subroutine warning

  !> Common error handling of bml. This function writes out a
  !! non-fatal warning message.
  !!
  !! In the future one could imagine something more like exceptions,
  !! in which the error gets passed up the call stack.
  !!
  !! @param file The filename in which the error occurred.
  !! @param line The line number in that file.
  !! @param message The error message.
  subroutine debug(file, line, message)

    character(len=*), intent(in) :: file, message
    integer, intent(in) :: line

    call bml_msg(file, line, "DEBUG", message)

  end subroutine debug

end module bml_error_m
