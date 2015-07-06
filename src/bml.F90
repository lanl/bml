!> \mainpage Basic Matrix Library (bml)
!!
!! \author
!! Nicolas Bock <nbock@lanl.gov>

!> \defgroup allocate_group Allocation and Deallocation Functions
!> \defgroup initialize_group Initialization Functions for Matrices

!> \copyright Los Alamos National Laboratory 2015

!> Main matrix library module.
module bml

  ! Put this first.
  use bml_type

  ! Add all of the other modules.
  use bml_add
  use bml_allocate
  use bml_convert
  use bml_error
  use bml_multiply
  use bml_print
  use bml_scale

end module bml
