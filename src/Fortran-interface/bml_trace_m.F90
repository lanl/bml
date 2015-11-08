!> Matrix trace.
module bml_trace_m

  implicit none
  private

  interface

     function bml_trace_C(a) bind(C, name="bml_trace")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       real(C_DOUBLE) :: bml_trace_C
     end function bml_trace_C

  end interface

  public :: bml_trace

contains

  !> Calculate the trace of a matrix.
  !!
  !! \f$ \leftarrow \mathrm{Tr} \left[ A \right] \f$
  !!
  !! \param a The matrix.
  function bml_trace(a) result(tr_a)

    use bml_types_m

    class(bml_matrix_t), intent(in) :: a
    double precision :: tr_a

    tr_a = bml_trace_C(a%ptr)

  end function bml_trace

end module bml_trace_m
