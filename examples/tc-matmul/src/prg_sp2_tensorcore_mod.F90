!> The SP2 Tensor Core module.
!! \ingroup PROGRESS
!!
!! \brief This subroutine implements SP2 density matrix
!!  purification algorithm using Tenso Core devices.
!!
module prg_sp2_tensorcore_mod

  use, intrinsic :: iso_c_binding !C interface

  implicit none
  public :: prg_sp2_tensorcore_f , prg_sp2_tensorcore_C

  interface

    subroutine prg_sp2_tensorcore_C(N,H,D,eps,bndfil,minsp2iter,maxsp2iter,&
         & sp2conv,idemtol,verbose) bind(C, name="prg_sp2_tensorcore")
      import :: C_PTR, C_INT, C_FLOAT, C_CHAR
      type(C_PTR), value :: D
      type(C_PTR), value :: H
      real(C_FLOAT), value, intent(in) :: eps, idemtol, bndfil
      integer(C_INT), value, intent(in) :: N, minsp2iter, maxsp2iter, verbose
      character(C_CHAR), intent(in)  :: sp2conv(*)
    end subroutine prg_sp2_tensorcore_C

  end interface

contains
 
  !> Calculates the density matrix from a Hamiltonian matrix by
  !! purification.
  !!
  !! \param N Number of orbitals (Size of the Hamiltonian).
  !! \param H Input Hamiltonian matrix.
  !! \param D Output density matrix.
  !! \param threshold Threshold for sparse matrix algebra.
  !! \param bndfil Band filling fraction.
  !! \param minsp2iter Minimum sp2 iterations.
  !! \param maxsp2iter Maximum SP2 iterations.
  !! \param sp2conv Convergence type.
  !! \param idemtol Idempotency tolerance.
  !! \param verbose A verbosity level. 
  subroutine prg_sp2_tensorcore_f(N,H,D,eps,bndfil,minsp2iter,maxsp2iter,&
       & sp2conv,idemtol,verbose)
    integer(C_INT), intent(in) :: N, minsp2iter, maxsp2iter, verbose
    real(C_DOUBLE), target, intent(inout) :: D(*)
    real(C_FLOAT), target, intent(in) :: H(*)
    real(C_FLOAT), intent(in) :: eps, bndfil, idemtol
    character(C_CHAR), intent(in) :: sp2conv

    !Call the interface
    call prg_sp2_tensorcore_C(N,c_loc(H),c_loc(D),eps,bndfil,minsp2iter,maxsp2iter,sp2conv,idemtol,verbose)

  end subroutine prg_sp2_tensorcore_f

end module prg_sp2_tensorcore_mod

