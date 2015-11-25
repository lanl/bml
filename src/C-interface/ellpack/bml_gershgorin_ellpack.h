#ifndef __BML_GERSHGORIN_ELLPACK_H
#define __BML_GERSHGORIN_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_gershgorin_ellpack(
    const bml_matrix_ellpack_t * A,
    double maxeval,
    double maxminusmin,
    const double threshold);

void bml_gershgorin_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    double maxeval,
    double maxminusmin,
    const double threshold);

void bml_gershgorin_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    double maxeval,
    double maxminusmin,
    const double threshold);

/*
void bml_gershgorin_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    double maxeval,
    double maxminusmin,
    const double threshold);

void bml_gershgorin_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    double maxeval,
    double maxminusmin,
    const double threshold);
*/

#endif
