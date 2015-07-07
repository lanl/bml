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
  use bml_add_m
  use bml_allocate_m
  use bml_convert_m
  use bml_error_m
  use bml_multiply_m
  use bml_print_m
  use bml_scale_m
  use bml_trace_m

end module bml
