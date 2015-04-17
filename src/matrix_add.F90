!> Matrix addition.
module matrix_add

  use matrix_add_dense

contains

  !> Add two matrices.
  !!
  !! \f$ C \leftarrow A+B \f$
  !!
  !! @param A Matrix \f$ A \f$.
  !! @param B Matrix \f$ B \f$.
  !! @param C Matrix \f$ C \f$.
  subroutine add (A, B, C)

    type(matrix_t), intent(in) :: A, B
    type(matrix_t), intent(inout) :: C

    select case(A%matrix_type)
    case(matrix_type_name_dense)
       select case(B%matrix_type)
       case(matrix_type_name_dense)
          call add_dense(A, B, C)
       case default
          write(*, *) "[add] matrix type mismatch"
          error stop
       end select
    case default
       write(*, *) "[add] not implemented"
       error stop
    end select

  end subroutine add

end module matrix_add
