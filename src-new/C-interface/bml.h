/** \mainpage Basic Matrix Library (bml)
 *
 * This library implements a common API for linear algebra and matrix
 * functions in C and Fortran. It offers several data structures for
 * matrix storage and algorithms. Currently the following matrix data
 * types are implemented:
 *   - dense
 *   - sparse (ELLPACK)
 *
 * \section fortran_usage Fortran Usage
 *
 * The use of this library is pretty straightforward. In the
 * application code, `use` the bml main module,
 *
 * \code{.F90}
 * use bml
 * \endcode
 *
 * A matrix is of type
 *
 * \code{.F90}
 * type(bml_matrix_t) :: a
 * \endcode
 *
 * There are two important things to note. First, although not
 * explicitly state in the above example, the matrix is not yet
 * allocated. Hence, the matrix needs to be allocated through an
 * allocation procedure with the desired type and precision,
 * e.g. dense:double, see the page on \ref allocate_group_Fortran
 * "allocation functions" for a complete list. For instance,
 *
 * \code{.F90}
 * call bml_zero_matrix(BML_MATRIX_DENSE, BML_PRECISION_DOUBLE, 100, a)
 * \endcode
 *
 * will allocate a dense, double-precision, \f$ 100 \times 100 \f$
 * matrix which is initialized to zero. Additional functions allocate
 * special matrices,
 *   - bml_allocate::bml_random_matrix Allocate and initialize a
 *     random matrix.
 *   - bml_allocate::bml_identity_matrix Allocate and initialize the
 *     identity matrix.
 *
 * A matrix is deallocated by calling
 *
 * \code{.F90}
 * call bml_deallocate(a)
 * \endcode
 *
 * \section C_usage C Usage
 *
 * In C, the following example code does the same as the above Fortran code:
 *
 * \code{.c}
 * #include <bml.h>
 *
 * bml_matrix_t *A = bml_zero_matrix(dense, single_precision, 100);
 * bml_deallocate(&A);
 * \endcode
 *
 * \author Jamaludin Mohd-Yusof <jamal@lanl.gov>
 * \author Nicolas Bock <nbock@lanl.gov>
 * \author Susan M. Mniszewski <smm@lanl.gov>
 *
 * \copyright Los Alamos National Laboratory 2015
 *
 * \todo
 * Add full support for dense matrix type for single precision.
 *
 * \defgroup allocate_group_C Allocation and Deallocation Functions (C interface)
 * \defgroup convert_group_C Converting between Matrix Formats (C interface)
 * \defgroup allocate_group_Fortran Allocation and Deallocation Functions (Fortran interface)
 * \defgroup convert_group_Fortran Converting between Matrix Formats (Fortran interface)
 */

/** \copyright Los Alamos National Laboratory 2015 */

/** \file */

#ifndef __BML_H
#define __BML_H

#include "bml_allocate.h"
#include "bml_convert.h"
#include "bml_logger.h"
#include "bml_utilities.h"

#endif
