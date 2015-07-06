!> \copyright Los Alamos National Laboratory 2015

!> Matrix addition.
module bml_add
  implicit none
contains

  !> Add two matrices.
  !!
  !! \f$ C \leftarrow \alpha A + \beta B \f$
  !!
  !! \param A Matrix \f$ A \f$
  !! \param B Matrix \f$ B \f$
  !! \param C Matrix \f$ C \f$
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add(A, B, C, alpha, beta)

    use bml_type_dense
    use bml_add_dense
    use bml_allocate
    use bml_error

    class(bml_matrix_t), allocatable, intent(in) :: A, B
    class(bml_matrix_t), allocatable, intent(inout) :: C
    double precision, optional :: alpha, beta

    if(.not. allocated(A) .or. .not. allocated(B)) then
       call error(__FILE__, __LINE__, "either A or B are not allocated")
    endif

    select type(A)
    type is(bml_matrix_dense_t)
       select type(B)
       type is(bml_matrix_dense_t)
          call deallocate_matrix(C)
          allocate(bml_matrix_dense_t::C)
          select type(C)
          type is(bml_matrix_dense_t)
             call add_dense(A, B, C, alpha, beta)
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

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! \param A Matrix A
  !! \param C Matrix C
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity(A, C, alpha, beta)

    use bml_type_dense
    use bml_add_dense
    use bml_error

    class(bml_matrix_t), allocatable, intent(in) :: A
    class(bml_matrix_t), allocatable, intent(out) :: C
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

    if(.not. allocated(A)) then
       call error(__FILE__, __LINE__, "A is not allocated")
    endif

  end subroutine add_identity

end module bml_add
