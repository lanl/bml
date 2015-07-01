!> @copyright Los Alamos National Laboratory 2015

!> Matrix multiplication.
module bml_multiply

  use bml_multiply_dense

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

    class(matrix_t), intent(in) :: A, B
    class(matrix_t), intent(inout) :: C
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

    select type(A)
    type is(matrix_dense_t)
       select type(B)
       type is(matrix_dense_t)
          select type(C)
          type is(matrix_dense_t)
             call multiply_dense(A, B, C)
          class default
             write(*, *) "[Multiply] C matrix type mismatch"
             error stop
          end select
       class default
          write(*, *) "[multiply] matrix type mismatch"
          error stop
       end select
    class default
       write(*, *) "[multiply] not implemented"
       error stop
    end select

  end subroutine multiply

end module bml_multiply
