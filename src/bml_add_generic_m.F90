!> \copyright Los Alamos National Laboratory 2015

!> Matrix addition.
module bml_add_BML_PRECISION_NAME_m

  implicit none

  private

  !> \addtogroup add_group
  !! @{

  !> Add two matrices.
  interface bml_add
     module procedure add_two_BML_PRECISION_NAME
     module procedure add_three_BML_PRECISION_NAME
  end interface bml_add

  !> Add identity matrix to a matrix.
  interface bml_add_identity
     module procedure add_identity_one_BML_PRECISION_NAME
     module procedure add_identity_two_BML_PRECISION_NAME
  end interface bml_add_identity
  !> @}

  public :: bml_add
  public :: bml_add_identity

contains

  !> Add two matrices.
  !!
  !! \f$ A \leftarrow \alpha A + \beta B \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param alpha Factor \f$ \alpha \f$
  !! \param a Matrix \f$ A \f$
  !! \param beta Factor \f$ \beta \f$
  !! \param b Matrix \f$ B \f$
  subroutine add_two_BML_PRECISION_NAME(alpha, a, beta, b)

    use bml_type_m
    use bml_type_dense_m
    use bml_add_dense_BML_PRECISION_NAME_m
    use bml_allocate_m
    use bml_error_m

    BML_REAL, intent(in) :: alpha
    class(bml_matrix_t), intent(inout) :: a
    BML_REAL, intent(in) :: beta
    class(bml_matrix_t), intent(in) :: b

    if(a%n /= b%n) then
       call bml_error(__FILE__, __LINE__, "matrix dimension mismatch")
    end if

    select type(a)
    type is(bml_matrix_dense_BML_PRECISION_NAME_t)
       select type(b)
       type is(bml_matrix_dense_BML_PRECISION_NAME_t)
          call add_two_dense(alpha, a, beta, b)
       class default
          call bml_error(__FILE__, __LINE__, "matrix type mismatch")
       end select
    class default
       call bml_error(__FILE__, __LINE__, "not implemented")
    end select

  end subroutine add_two_BML_PRECISION_NAME

  !> Add two matrices.
  !!
  !! \f$ C \leftarrow \alpha A + \beta B \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param alpha Factor \f$ \alpha \f$
  !! \param a Matrix \f$ A \f$
  !! \param beta Factor \f$ \beta \f$
  !! \param b Matrix \f$ B \f$
  !! \param c Matrix \f$ C \f$
  subroutine add_three_BML_PRECISION_NAME(alpha, a, beta, b, c)

    use bml_type_m
    use bml_type_dense_m
    use bml_add_dense_BML_PRECISION_NAME_m
    use bml_allocate_m
    use bml_error_m

    BML_REAL, intent(in) :: alpha
    class(bml_matrix_t), intent(in) :: a
    BML_REAL, intent(in) :: beta
    class(bml_matrix_t), intent(in) :: b
    class(bml_matrix_t), allocatable, intent(inout) :: c

    if(a%n /= b%n) then
       call bml_error(__FILE__, __LINE__, "matrix dimension mismatch")
    end if

    select type(a)
    type is(bml_matrix_dense_BML_PRECISION_NAME_t)
       select type(b)
       type is(bml_matrix_dense_BML_PRECISION_NAME_t)
          call bml_allocate(BML_MATRIX_DENSE, a%n, c, BML_PRECISION_BML_PRECISION_NAME)
          select type(c)
          type is(bml_matrix_dense_BML_PRECISION_NAME_t)
             call add_three_dense(a, b, c, alpha, beta)
          class default
             call bml_error(__FILE__, __LINE__, "C matrix type mismatch")
          end select
       class default
          call bml_error(__FILE__, __LINE__, "matrix type mismatch")
       end select
    class default
       call bml_error(__FILE__, __LINE__, "not implemented")
    end select

  end subroutine add_three_BML_PRECISION_NAME

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param a Matrix A
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_one_BML_PRECISION_NAME(a, alpha, beta)

    use bml_type_m
    use bml_type_dense_m
    use bml_add_dense_BML_PRECISION_NAME_m
    use bml_allocate_m
    use bml_error_m

    class(bml_matrix_t), intent(inout) :: a
    BML_REAL, intent(in) :: alpha, beta

    select type(a)
    type is(bml_matrix_dense_BML_PRECISION_NAME_t)
       call add_identity_self_dense(a, alpha, beta)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine add_identity_one_BML_PRECISION_NAME

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param alpha Factor \f$ \alpha \f$
  !! \param a Matrix A
  !! \param c Matrix C
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_two_BML_PRECISION_NAME(alpha, a, c, beta)

    use bml_type_m
    use bml_type_dense_m
    use bml_add_dense_BML_PRECISION_NAME_m
    use bml_allocate_m
    use bml_error_m

    BML_REAL, intent(in) :: alpha
    class(bml_matrix_t), intent(in) :: a
    class(bml_matrix_t), allocatable, intent(out) :: c
    BML_REAL, intent(in) :: beta

    select type(a)
    type is(bml_matrix_dense_BML_PRECISION_NAME_t)
       call bml_allocate(BML_MATRIX_DENSE, a%n, c, BML_PRECISION_BML_PRECISION_NAME)
       select type(c)
       type is(bml_matrix_dense_BML_PRECISION_NAME_t)
          call add_identity_two_dense(alpha, a, c, beta)
       end select
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine add_identity_two_BML_PRECISION_NAME

end module bml_add_BML_PRECISION_NAME_m
