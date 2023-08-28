#ifndef __BML_TYPES_CSR_H
#define __BML_TYPES_CSR_H

#include "../bml_types.h"

#ifdef BML_USE_MPI
#include <mpi.h>
#endif

#define INIT_ROW_SPACE 10
#define EXPAND_FACT 1.3
#define INIT_SLOT_STORAGE_SIZE 5

typedef enum INSERTMODE
{ INSERT, ADD } INSERTMODE;
/** csr matrix type. */
/** slot type for hash table */
typedef struct csr_hash_slot_t
{
    /** pointer to a next entry in this slot. */
    struct csr_hash_slot_t *link;
    /** key in a pair (key,value) */
    int key;
    /** value in a pair (key, value) */
    int value;
} csr_hash_slot_t;
/** hash table for global-to-local mapping of csr matrix row indexes */
typedef struct csr_row_index_hash_t
{
    /** array of a pointer array to struct Slot  */
    csr_hash_slot_t **Slots_;
    /** size of array Slots. */
    int space_;
    int space_minus1_;
    /** number of pairs in the table. */
    int size_;

    /** pointer to slot in memory */
    csr_hash_slot_t *slot_ptr_;
    /** storage for slots */
    csr_hash_slot_t *slot_storage_;
    int slot_storage_space_;
    /** size of slot_storage */
    int capacity_;
} csr_row_index_hash_t;
/** sparse row of csr matrix */
typedef struct csr_sparse_row_t
{
    /** The real precision. */
    bml_matrix_precision_t row_precision;
    /** number of columns/ entries */
    int NNZ_;
    /** row entries */
    void *vals_;
    /** (global) column indexes on row */
    int *cols_;
    /** memory allocation for nonzero entries */
    int alloc_size_;
} csr_sparse_row_t;
struct bml_matrix_csr_t
{
    /** The matrix type identifier. */
    bml_matrix_type_t matrix_type;
    /** The real precision. */
    bml_matrix_precision_t matrix_precision;
    /** The distribution mode. **/
    bml_distribution_mode_t distribution_mode;

    /** The number of rows. */
    int N_;
    /** The Max. number of nonzeros per row. */
    int NZMAX_;
    /** The total number of nonzeros of the matrix */
    int TOTNNZ_;
    /** The array of local variable's global index. */
    int *lvarsgid_;
    /** Hash table for holding global, local index pairs */
    csr_row_index_hash_t *table_;
    /** The matrix data */
    csr_sparse_row_t **data_;

    /** The domain decomposition when running in parallel. */
    bml_domain_t *domain;
#ifdef BML_USE_MPI
    /** Buffer for communications */
    void *buffer;
    int *nnz_buffer;
    int *cols_buffer;
    /** request field for MPI communications*/
    MPI_Request req[3];
#endif
};
typedef struct bml_matrix_csr_t bml_matrix_csr_t;

/****** some accessor functions ****/
/** hash table **/
#define hash_key_index(key, space) ( (key) & (space))
#define hash_table_size(table) ((table)->size_)
/** csr row **/
#define csr_row_NNZ(csr_row) ((csr_row)->NNZ_)
#define csr_row_alloc_size(csr_row) ((csr_row)->alloc_size_)
/** csr matrix **/
#define csr_matrix_N(csr_matrix) ((csr_matrix)->N_)
#define csr_matrix_NZMAX(csr_matrix) ((csr_matrix)->NZMAX_)
#define csr_matrix_TOTNNZ(csr_matrix) ((csr_matrix)->TOTNNZ_)
#define csr_matrix_type(csr_matrix) ((csr_matrix)->matrix_type)
#define csr_matrix_precision(csr_matrix) ((csr_matrix)->matrix_precision)
#define csr_matrix_distribution_mode(csr_matrix) ((csr_matrix)->distribution_mode)

#endif
