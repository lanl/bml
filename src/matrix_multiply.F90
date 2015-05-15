!> @copyright Los Alamos National Laboratory 2015

!> Matrix multiplication.
module matrix_multiply

  use matrix_multiply_dense

contains

  !> Multiply two matrices.
  !!
  !! \f$ C \leftarrow \alpha A \times B + \beta C \f$
  !!
  !! @param A Matrix \f$ A \f$.
  !! @param B Matrix \f$ B \f$.
  !! @param C Matrix \f$ C \f$.
  !! @param alpha The factor \f$ \alpha \f$.
  !! @param beta The factor \f$ \beta \f$.
  subroutine multiply (A, B, C, alpha, beta)

    type(matrix_t), intent(in) :: A, B
    type(matrix_t), intent(inout) :: C
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

    select case(A%matrix_type)
    case(matrix_type_name_dense)
       select case(B%matrix_type)
       case(matrix_type_name_dense)
          call multiply_dense(A, B, C)
       case default
          write(*, *) "[multiply] matrix type mismatch"
          error stop
       end select
    case default
       write(*, *) "[multiply] not implemented"
       error stop
    end select

  end subroutine multiply

end module matrix_multiply
