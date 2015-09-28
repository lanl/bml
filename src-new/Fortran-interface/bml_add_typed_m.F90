submodule (bml_add_m) bml_add_MATRIX_TYPE_m

  implicit none

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
  subroutine add_two_MATRIX_TYPE(alpha, a, beta, b)
  end subroutine add_two_MATRIX_TYPE

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
  subroutine add_three_MATRIX_TYPE(alpha, a, beta, b, c)
  end subroutine add_three_MATRIX_TYPE

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
  subroutine add_identity_one_MATRIX_TYPE(a, alpha, beta)
  end subroutine add_identity_one_MATRIX_TYPE

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
  subroutine add_identity_two_MATRIX_TYPE(alpha, a, c, beta)
  end subroutine add_identity_two_MATRIX_TYPE

end submodule bml_add_MATRIX_TYPE_m
