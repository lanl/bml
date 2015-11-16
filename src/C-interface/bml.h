/** \mainpage Basic Matrix Library (bml)
 *
 * This library implements a common API for linear algebra and matrix
 * functions in C and Fortran. It offers several data structures for
 * matrix storage and algorithms. Currently the following matrix data
 * types are implemented:
 *   - dense
 *   - ellpack (sparse)
 *   - csr (sparse)
 *
 * \section usage_examples Usage Examples
 *
 * Usage examples can be found here:
 *   - \ref fortran_usage "Fortran Usage"
 *   - \ref C_usage "C Usage"
 *
 * \section developers Modifying the library itself
 *
 * If you are interested in modifying the library code itself, please
 * have a look at the
 * \ref developer_documentation "Developer Documentation".
 *
 * \section planned_features Planned Features
 *
 * We are planning to eventually support different matrix types and
 * matrix operations on a variety of hardware platforms. For details,
 * please have a look at our \ref future_plans "future plans".
 *
 * \author Christian Negre <cnegre@lanl.gov>
 * \author Jamaludin Mohd-Yusof <jamal@lanl.gov>
 * \author Nicolas Bock <nbock@lanl.gov>
 * \author Susan M. Mniszewski <smm@lanl.gov>
 *
 * \copyright Los Alamos National Laboratory 2015
 *
 * \defgroup allocate_group_C Allocation and Deallocation Functions (C interface)
 * \defgroup add_group_C Add Functions (C interface)
 * \defgroup convert_group_C Converting between Matrix Formats (C interface)
 * \defgroup allocate_group_Fortran Allocation and Deallocation Functions (Fortran interface)
 * \defgroup add_group_Fortran Add Functions (Fortran interface)
 * \defgroup convert_group_Fortran Converting between Matrix Formats (Fortran interface)
 */

/** \page future_plans Future Plans
 *
 * \section planned_type Matrix Types
 *
 * Support types:
 *     - bml_matrix_t
 *     - Colinear
 *     - Noncolinear
 *     - Blocked Bloch Matrix
 *
 * \section planned_precisions Precisions
 *
 * The bml supports the following precisions:
 *     - logical (for matrix masks)
 *     - single real
 *     - double real
 *     - single complex
 *     - double complex
 *
 * \section planned_functions Functions
 *
 * The library supports the following matrix operations:
 *     - Format Conversion
 *         - bml_convert::bml_convert_from_dense
 *         - bml_convert::bml_convert_to_dense
 *         - bml_convert::bml_convert
 *     - Masking
 *         - Masked operations (restricted to a subgraph)
 *     - Addition
 *         - \f$ \alpha A + \beta B \f$: bml_add::bml_add
 *         - \f$ \alpha A + \beta \f$: bml_add::bml_add_identity
 *     - Copy
 *         - \f$ B \leftarrow A \f$: bml_copy::bml_copy
 *     - Diagonalize
 *         - bml_diagonalize::bml_diagonalize
 *     - Introspection
 *         - bml_introspection::bml_get_type
 *         - bml_introspection::bml_get_size
 *         - bml_introspection::bml_get_bandwidth
 *         - bml_introspection::bml_get_spectral_range
 *         - bml_introspection::bml_get_HOMO_LUMO
 *     - Matrix manipulation:
 *         - bml_get::bml_get
 *         - bml_get::bml_get_rows
 *         - bml_set::bml_set
 *         - bml_set::bml_set_rows
 *     - Multiplication
 *         - \f$ \alpha A \times B + \beta C \f$: bml_multiply::bml_multiply
 *     - Printing
 *         - bml_utilities::bml_print_matrix
 *     - Scaling
 *         - \f$ A \leftarrow \alpha A \f$: bml_scale::bml_scale_one
 *         - \f$ B \leftarrow \alpha A \f$: bml_scale::bml_scale_two
 *     - Matrix trace
 *         - \f$ \mathrm{Tr} [ A ] \f$: bml_trace::bml_trace
 *         - \f$ \mathrm{Tr} [ A B ] \f$: bml_trace::bml_product_trace
 *     - Matrix norm
 *         - 2-norm
 *         - Frobenius norm
 *     - Matrix transpose
 *         - bml_transpose::bml_transpose
 *     - Matrix commutator/anticommutator
 *         - bml_commutator::bml_commutator
 *         - bml_commutator::bml_anticommutator
 *
 * Back to the \ref index "main page".
 */

/** \page C_usage C Usage
 *
 * In C, the following example code does the same as the above Fortran code:
 *
 * \code{.c}
 * #include <bml.h>
 *
 * bml_matrix_t *A = bml_zero_matrix(dense, single_real, 100);
 * bml_deallocate(&A);
 * \endcode
 *
 * Back to the \ref index "main page".
 */

/** \page fortran_usage Fortran Usage
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
 * Back to the \ref index "main page".
 */

/** \page developer_documentation Developer Documentation
 *
 * \section workflow Developer Suggested Workflow
 *
 * We try to preserve a linear history in our main (master)
 * branch. Instead of pulling (i.e. merging), we suggest you use:
 *
 *     $ git pull --rebase
 *
 * And then
 *
 *     $ git push
 *
 * To push your changes back to the server.
 *
 * \section coding_style Coding Style
 *
 * Please indent your C code using
 *
 *     $ indent -gnu -nut -i4 -bli0
 *
 * Back to the \ref index "main page".
 */

/** \copyright Los Alamos National Laboratory 2015 */

/** \file */

#ifndef __BML_H
#define __BML_H

#include "bml_add.h"
#include "bml_allocate.h"
#include "bml_convert.h"
#include "bml_copy.h"
#include "bml_diagonalize.h"
#include "bml_export.h"
#include "bml_import.h"
#include "bml_logger.h"
#include "bml_multiply.h"
#include "bml_scale.h"
#include "bml_threshold.h"
#include "bml_trace.h"
#include "bml_transpose.h"
#include "bml_utilities.h"

#endif
