#include "../macros.h"
#include "../typed.h"
#include "../bml_introspection.h"
#include "bml_setters_ellpack.h"
#include "bml_types_ellpack.h"

#include <complex.h>

void TYPED_FUNC(
    bml_set_ellpack) (
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value)
{
}

void TYPED_FUNC(
    bml_set_row_ellpack) (
    bml_matrix_ellpack_t * A,
    const int i,
    const REAL_T * row)
{
}
