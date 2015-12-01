module bml_c_interface_m
  use, intrinsic :: iso_c_binding
  implicit none

  interface
    function bml_copy_new_C(a) bind(C, name="bml_copy_new")
      import :: C_PTR
      type(C_PTR), value, intent(in) :: a
      type(C_PTR) :: bml_copy_new_C
    end function bml_copy_new_C

    subroutine bml_deallocate_C(a) bind(C, name="bml_deallocate")
      import :: C_PTR
      type(C_PTR) :: a
    end subroutine bml_deallocate_C

  end interface

end module bml_c_interface_m
