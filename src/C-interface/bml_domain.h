#ifndef __BML_DOMAIN_H
#define __BML_DOMAIN_H

#include "bml_types.h"

bml_domain_t *bml_default_domain(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

void bml_deallocate_domain(
    bml_domain_t * D);

void bml_copy_domain(
    bml_domain_t * A,
    bml_domain_t * B);

void bml_update_domain(
    bml_domain_t * A_domain,
    int *localPartMin,
    int *localPartMax,
    int *nnodesInPart);

#endif
