/** \file */

#ifndef __BML_GERSHGORIN_H
#define __BML_GERSHGORIN_H

#include "bml_types.h"

// Calculate Gershgorin bounds
void bml_gershgorin(
    const bml_matrix_t * A,
    double maxeval;
    double maxminusmin,
    const double threshold);

#endif
