!> \copyright Los Alamos National Laboratory 2015

!> Matrix addition.
module bml_add_m
  implicit none

  interface add_identity
     module procedure add_identity_two
     module procedure add_identity_self
  end interface add_identity

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
    use bml_allocate_m
    use bml_error_m

    class(bml_matrix_t), allocatable, intent(in) :: A, B
    class(bml_matrix_t), allocatable, intent(inout) :: C
    double precision, optional :: alpha, beta

    if(.not. allocated(A) .or. .not. allocated(B)) then
       call error(__FILE__, __LINE__, "either A or B are not allocated")
    end if

    select type(A)
    type is(bml_matrix_dense_t)
       select type(B)
       type is(bml_matrix_dense_t)
          call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, A%N, C)
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
  subroutine add_identity_two(A, C, alpha, beta)

    use bml_type_dense
    use bml_add_dense
    use bml_allocate_m
    use bml_error_m

    class(bml_matrix_t), allocatable, intent(in) :: A
    class(bml_matrix_t), allocatable, intent(out) :: C
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

    if(.not. allocated(A)) then
       call error(__FILE__, __LINE__, "A is not allocated")
    end if

    select type(A)
    type is(bml_matrix_dense_t)
       select type(C)
          type is(bml_matrix_dense_t)
             call add_identity_two_dense(A, C, alpha, beta)
          end select
    class default
       call error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine add_identity_two

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! \param A Matrix A
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_self(A, alpha, beta)

    use bml_type_dense
    use bml_add_dense
    use bml_allocate_m
    use bml_error_m

    class(bml_matrix_t), allocatable, intent(inout) :: A
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

    if(.not. allocated(A)) then
       call error(__FILE__, __LINE__, "A is not allocated")
    end if

    select type(A)
    type is(bml_matrix_dense_t)
       call add_identity_self_dense(A, alpha, beta)
    class default
       call error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine add_identity_self

end module bml_add_m
