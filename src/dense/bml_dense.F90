!> \copyright Los Alamos National Laboratory 2015

!> Main dense matrix library module.
module bml_dense

  ! Put this first.
  use bml_type_dense

  ! Add all of the other modules.
  use bml_add_dense
  use bml_convert_dense
  use bml_allocate_dense
  use bml_multiply_dense
  use bml_print_dense
  use bml_scale_dense

end module bml_dense
