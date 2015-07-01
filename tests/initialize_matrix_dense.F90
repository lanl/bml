program test

  use bml

  implicit none

  integer, parameter :: N = 12
  class(bml_matrix_t), allocatable :: A

  call zero_matrix("dense", N, A)

end program test
