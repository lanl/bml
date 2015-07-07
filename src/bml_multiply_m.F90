!> \copyright Los Alamos National Laboratory 2015

!> Matrix multiplication.
module bml_multiply_m
  implicit none
contains

  !> Multiply two matrices.
  !!
  !! \f$ C \leftarrow \alpha A \times B + \beta C \f$
  !!
  !! \param A Matrix \f$ A \f$.
  !! \param B Matrix \f$ B \f$.
  !! \param C Matrix \f$ C \f$.
  !! \param alpha The factor \f$ \alpha \f$.
  !! \param beta The factor \f$ \beta \f$.
  subroutine multiply (A, B, C, alpha, beta)

    use bml_type_dense

    use bml_allocate_m
    use bml_error_m
    use bml_multiply_dense

    class(bml_matrix_t), allocatable, intent(in) :: A, B
    class(bml_matrix_t), allocatable, intent(inout) :: C
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

    if(.not. allocated(A) .or. .not. allocated(B)) then
       call error(__FILE__, __LINE__, "either A or B are not allocated")
    endif

    select type(A)
    type is(bml_matrix_dense_t)
       select type(B)
       type is(bml_matrix_dense_t)
          call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, A%N, C)
          select type(C)
          type is(bml_matrix_dense_t)
             call multiply_dense(A, B, C)
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
