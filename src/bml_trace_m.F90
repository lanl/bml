!> \copyright Los Alamos National Laboratory 2015

!> Matrix trace.
module bml_trace_m
  implicit none
contains

  !> Calculate the trace of a matrix.
  !!
  !! \f$ \leftarrow \mathrm{Tr} \left[ A \right] \f$
  !!
  !! \param A The matrix.
  function trace(A) result(trA)

    use bml_type_dense
    use bml_trace_dense_m
    use bml_error_m

    class(bml_matrix_t), allocatable, intent(in) :: A
    double precision :: trA

    trA = 0
    if(.not. allocated(A)) return

    select type(A)
    type is(bml_matrix_dense_t)
       !trA = trace_dense(A)
    class default
       call error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end function trace

end module bml_trace_m
