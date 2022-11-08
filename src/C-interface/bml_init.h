/** \file */

#ifndef __BML_INIT_H
#define __BML_INIT_H

#include "bml_types.h"

#ifdef BML_USE_MPI
#include <mpi.h>
#endif

void bml_init(
#ifdef BML_USE_MPI
    MPI_Comm comm
#endif
    );

void bml_initF(
    int fcomm);

#endif
