!> \copyright Los Alamos National Laboratory 2015

!> Matrix multiplication.
module bml_multiply_m
  implicit none
contains

  !> Multiply two matrices.
  !!
  !! \ingroup multiply_group
  !!
  !! \f$ C \leftarrow \alpha A \times B + \beta C \f$
  !!
  !! The optional scaling factors \f$ \alpha \f$ and \f$ \beta \f$
  !! default to \f$ \alpha = 1 \f$ and \f$ \beta = 0 \f$.
  !!
  !! \param A Matrix \f$ A \f$.
  !! \param B Matrix \f$ B \f$.
  !! \param C Matrix \f$ C \f$.
  !! \param alpha The factor \f$ \alpha \f$.
  !! \param beta The factor \f$ \beta \f$.
  subroutine multiply(A, B, C, alpha, beta)

    use bml_type_dense_m
    use bml_allocate_m
    use bml_error_m
    use bml_multiply_dense_m

    class(bml_matrix_t), intent(in) :: A, B
    class(bml_matrix_t), allocatable, intent(inout) :: C
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

    double precision :: alpha_, beta_

    if(present(alpha)) then
       alpha_ = alpha
    else
       alpha_ = 1
    end if
    if(present(beta)) then
       beta_ = beta
    else
       beta_ = 0
    end if

    if(allocated(C)) then
       if(A%N /= C%N) then
          call error(__FILE__, __LINE__, "matrix dimension mismatch")
       end if
    end if

    select type(A)
    type is(bml_matrix_dense_double_t)
       select type(B)
       type is(bml_matrix_dense_double_t)
          if(.not. allocated(C)) then
             call allocate_matrix(BML_MATRIX_DENSE, A%N, C)
          end if
          select type(C)
          type is(bml_matrix_dense_double_t)
             call multiply_dense(A, B, C, alpha_, beta_)
          class default
             call error(__FILE__, __LINE__, "C matrix type mismatch")
          end select
       class default
          call error (__FILE__, __LINE__, "matrix type mismatch")
       end select
    class default
       call error(__FILE__, __LINE__, "not implemented")
    end select

  end subroutine multiply

end module bml_multiply_m
