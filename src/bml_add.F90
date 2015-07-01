!> @copyright Los Alamos National Laboratory 2015

!> Matrix addition.
module bml_add

  use bml_add_dense
  use bml_error

contains

  !> Add two matrices.
  !!
  !! \f$ C \leftarrow A+B \f$
  !!
  !! @param A Matrix \f$ A \f$.
  !! @param B Matrix \f$ B \f$.
  !! @param C Matrix \f$ C \f$.
  subroutine add (A, B, C)

    class(bml_matrix_t), allocatable, intent(in) :: A, B
    class(bml_matrix_t), allocatable, intent(inout) :: C

    if(.not. allocated(A) .or. .not. allocated(B)) then
       call error(__FILE__, __LINE__, "either A or B are not allocated")
    endif

    select type(A)
    type is(bml_matrix_dense_t)
       select type(B)
       type is(bml_matrix_dense_t)
          if(.not. allocated(C)) then
             allocate(bml_matrix_dense_t::C)
          endif
          select type(C)
          type is(bml_matrix_dense_t)
             call add_dense(A, B, C)
          class default
             call error(__FILE__, __LINE__, "C matrix type mismatch")
          end select
       class default
          call error(__FILE__, __LINE__, "matrix type mismatch")
       end select
    class default
       call error(__FILE__, __LINE__, "not implemented")
    end select

  end subroutine add

end module bml_add
