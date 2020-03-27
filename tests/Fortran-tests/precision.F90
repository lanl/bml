
!> Presicion module
!! \brief Simple and double precision definition.
module prec

  implicit none

  private

  integer, parameter, public :: sp = kind(1.0)
  integer, parameter, public :: dp = kind(1.d0)

end module prec
