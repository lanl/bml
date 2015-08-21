!> \copyright Los Alamos National Laboratory 2015

!> Matrix addition.
module bml_add_m

  implicit none

  !> \addtogroup add_group
  !! @{

  !> Add two matrices.
  interface add
     module procedure add_two
     module procedure add_three
  end interface add

  !> Add identity matrix to a matrix.
  interface add_identity
     module procedure add_identity_one
     module procedure add_identity_two
  end interface add_identity
  !> @}

contains

  !> Add two matrices.
  !!
  !! \f$ A \leftarrow \alpha A + \beta B \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param A Matrix \f$ A \f$
  !! \param B Matrix \f$ B \f$
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_two(A, B, alpha, beta)

    use bml_type_dense_m
    use bml_add_dense_m
    use bml_allocate_m
    use bml_error_m

    class(bml_matrix_t), intent(inout) :: A
    class(bml_matrix_t), intent(in) :: B
    double precision, optional :: alpha, beta

    double precision :: alpha_, beta_

    if(A%N /= B%N) then
       call error(__FILE__, __LINE__, "matrix dimension mismatch")
    end if

    if(present(alpha)) then
       alpha_ = alpha
    else
       alpha_ = 1
    end if
    if(present(beta)) then
       beta_ = beta
    else
       beta_ = 1
    end if

    select type(A)
    type is(bml_matrix_dense_t)
       select type(B)
       type is(bml_matrix_dense_t)
          call add_two_dense(A, B, alpha_, beta_)
       class default
          call error(__FILE__, __LINE__, "matrix type mismatch")
       end select
    class default
       call error(__FILE__, __LINE__, "not implemented")
    end select

  end subroutine add_two

  !> Add two matrices.
  !!
  !! \f$ C \leftarrow \alpha A + \beta B \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param A Matrix \f$ A \f$
  !! \param B Matrix \f$ B \f$
  !! \param C Matrix \f$ C \f$
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_three(A, B, C, alpha, beta)

    use bml_type_dense_m
    use bml_add_dense_m
    use bml_allocate_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: A, B
    class(bml_matrix_t), allocatable, intent(inout) :: C
    double precision, optional :: alpha, beta

    double precision :: alpha_, beta_

    if(A%N /= B%N) then
       call error(__FILE__, __LINE__, "matrix dimension mismatch")
    end if

    if(present(alpha)) then
       alpha_ = alpha
    else
       alpha_ = 1
    end if
    if(present(beta)) then
       beta_ = beta
    else
       beta_ = 1
    end if

    select type(A)
    type is(bml_matrix_dense_t)
       select type(B)
       type is(bml_matrix_dense_t)
          call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, A%N, C)
          select type(C)
          type is(bml_matrix_dense_t)
             call add_three_dense(A, B, C, alpha_, beta_)
          class default
             call error(__FILE__, __LINE__, "C matrix type mismatch")
          end select
       class default
          call error(__FILE__, __LINE__, "matrix type mismatch")
       end select
    class default
       call error(__FILE__, __LINE__, "not implemented")
    end select

  end subroutine add_three

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param A Matrix A
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_one(A, alpha, beta)

    use bml_type_dense_m
    use bml_add_dense_m
    use bml_allocate_m
    use bml_error_m

    class(bml_matrix_t), intent(inout) :: A
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
       beta_ = 1
    end if

    select type(A)
    type is(bml_matrix_dense_t)
       call add_identity_self_dense(A, alpha, beta)
    class default
       call error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine add_identity_one

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param A Matrix A
  !! \param C Matrix C
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_two(A, C, alpha, beta)

    use bml_type_dense_m
    use bml_add_dense_m
    use bml_allocate_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: A
    class(bml_matrix_t), allocatable, intent(out) :: C
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
       beta_ = 1
    end if

    select type(A)
    type is(bml_matrix_dense_t)
       call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, A%N, C)
       select type(C)
       type is(bml_matrix_dense_t)
          call add_identity_two_dense(A, C, alpha_, beta_)
       end select
    class default
       call error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine add_identity_two

end module bml_add_m
