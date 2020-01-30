#include "bml_allocate.h"
#include "bml_allocate_csr.h"
#include "bml_logger.h"
#include "bml_parallel.h"
#include "bml_types.h"
#include "bml_types_csr.h"

#include <stdio.h>
#include <math.h>

/** allocate hash table.
 *
 * \ingroup allocate_group
 *
 * \param tsize - the initial hash table size.
 */
csr_row_index_hash_t *csr_noinit_table(
    const int alloc_size)
{  
  int i, lwr;

  static int powers[]    = {  9,  10,  11,  12,  13,   14,   15,   16,    17,    18};
  static int powersof2[] = {512,1024,2048,4096,8192,16384,32768,65536,131072,262144}; 
  for (i = 1; lwr = (powersof2[i]*2)/3, lwr < alloc_size; i++);
  const int space = powersof2[i-1];
  const int space_power_minus1= pow(2,powers[i-1])-1;
  
  /** create table object */
  csr_row_index_hash_t * table = 
      bml_noinit_allocate_memory(sizeof(csr_row_index_hash_t));
  
  /** allocate the space for member variables */
  table->size_ = 0;
  table->space_ = space;
  table->capacity_= space;
  table->space_power_minus1_ = space_power_minus1;
  /** allocate array of slot pointers and initialize slot pointers to NULL */
  table->Slots_ = 
      bml_noinit_allocate_memory(sizeof(csr_hash_slot_t *) * space);     

  for (i = 0; i < space; i++) 
  {
    table->Slots_[i] = NULL;
  }
  
  /** Allocate memory for storing data */
  table->slot_storage_space_ = INIT_SLOT_STORAGE_SIZE;
  table->slot_storage_ =
      bml_noinit_allocate_memory(sizeof(csr_hash_slot_t *) * table->slot_storage_space_); 
  csr_hash_slot_t* newstorage =
      bml_noinit_allocate_memory(sizeof(csr_hash_slot_t) * table->capacity_);
  
  table->slot_storage_[0] = newstorage;
  /** initialize slot_ptr_ */
  table->slot_ptr_ = newstorage;
  
  return table;
}

/** Deallocate hash table.
 *
 * \ingroup allocate_group
 *
 * \param table - the hash table.
 */
void
csr_deallocate_table(
    csr_row_index_hash_t * table)
{
    int tsize = table -> size_;
    /** delete allocated slots */
    for(int i=0; i<tsize; i++)
    {
       bml_free_memory((table -> slot_storage_)[i]);
    }
    bml_free_memory(table -> slot_storage_);
    bml_free_memory(table -> Slots_);
    bml_free_memory(table);
}

/** Deallocate csr row.
 *
 * \ingroup allocate_group
 *
 * \param row - the csr row.
 */
void
csr_deallocate_row(
    csr_sparse_row_t * row)
{
    bml_free_memory(row -> vals_);
    bml_free_memory(row -> cols_);
    bml_free_memory(row);
}

/** Deallocate a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void
bml_deallocate_csr(
    bml_matrix_csr_t * A)
{
    csr_deallocate_table(A->table_);    
    /** deallocate row data */
    const int n = A->N_;
    for(int i=0; i<n; i++)
    {
       csr_deallocate_row((A->data_)[i]);
    }
    bml_free_memory(A->data_);
    bml_free_memory(A->lvarsgid_);
    bml_free_memory(A);    
}

/** Clear a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void
bml_clear_csr(
    bml_matrix_csr_t * A)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_clear_csr_single_real(A);
            break;
        case double_real:
            bml_clear_csr_double_real(A);
            break;
        case single_complex:
            bml_clear_csr_single_complex(A);
            break;
        case double_complex:
            bml_clear_csr_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Allocate the csr matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The maximum number of non-zeros per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_csr_t *
bml_noinit_matrix_csr(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode)
{
    bml_matrix_csr_t *A = NULL;

    switch (matrix_precision)
    {
        case single_real:
            A = bml_noinit_matrix_csr_single_real(N,M, distrib_mode);
            break;
        case double_real:
            A = bml_noinit_matrix_csr_double_real(N,M, distrib_mode);
            break;
        case single_complex:
            A = bml_noinit_matrix_csr_single_complex(N,M, distrib_mode);
            break;
        case double_complex:
            A = bml_noinit_matrix_csr_double_complex(N,M, distrib_mode);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return A;
}

/** Allocate the zero matrix. (Currently assumes sequential case only)
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_csr_t *
bml_zero_matrix_csr(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,    
    const bml_distribution_mode_t distrib_mode)
{
    bml_matrix_csr_t *A = NULL;

    switch (matrix_precision)
    {
        case single_real:
            A = bml_zero_matrix_csr_single_real(N,M, distrib_mode);
            break;
        case double_real:
            A = bml_zero_matrix_csr_double_real(N,M, distrib_mode);
            break;
        case single_complex:
            A = bml_zero_matrix_csr_single_complex(N,M, distrib_mode);
            break;
        case double_complex:
            A = bml_zero_matrix_csr_double_complex(N,M, distrib_mode);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return A;
}

/** Allocate a banded random matrix. (Currently assumes sequential case only)
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
 
bml_matrix_csr_t *
bml_banded_matrix_csr(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_banded_matrix_csr_single_real(N, M, distrib_mode);
            break;
        case double_real:
            return bml_banded_matrix_csr_double_real(N, M, distrib_mode);
            break;
        case single_complex:
            return bml_banded_matrix_csr_single_complex(N, M,
                                                            distrib_mode);
            break;
        case double_complex:
            return bml_banded_matrix_csr_double_complex(N, M,
                                                            distrib_mode);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

/** Allocate a random matrix. (Currently assumes sequential case only)
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */

bml_matrix_csr_t *
bml_random_matrix_csr(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_random_matrix_csr_single_real(N, M, distrib_mode);
            break;
        case double_real:
            return bml_random_matrix_csr_double_real(N, M, distrib_mode);
            break;
        case single_complex:
            return bml_random_matrix_csr_single_complex(N, M,
                                                            distrib_mode);
            break;
        case double_complex:
            return bml_random_matrix_csr_double_complex(N, M,
                                                            distrib_mode);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

/** Allocate the identity matrix. (Currently assumes sequential case only)
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_csr_t *
bml_identity_matrix_csr(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_identity_matrix_csr_single_real(N,M,
                                                           distrib_mode);
            break;
        case double_real:
            return bml_identity_matrix_csr_double_real(N,M,
                                                           distrib_mode);
            break;
        case single_complex:
            return bml_identity_matrix_csr_single_complex(N,M,
                                                              distrib_mode);
            break;
        case double_complex:
            return bml_identity_matrix_csr_double_complex(N,M,
                                                              distrib_mode);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

/** Update the csr matrix domain.
 *
 * \ingroup allocate_group
 *
 * \param A Matrix with domain
 * \param localPartMin first part on each rank
 * \param localPartMax last part on each rank
 * \param nnodesInPart number of nodes per part
 */
void
bml_update_domain_csr(
    bml_matrix_csr_t * A,
    int *localPartMin,
    int *localPartMax,
    int *nnodesInPart)
{
    LOG_ERROR("bml_update_domain_csr not implemented\n");
}
