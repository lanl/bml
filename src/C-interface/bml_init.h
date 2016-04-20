/** \file */

#ifndef __BML_INIT_H
#define __BML_INIT_H

#include "bml_types.h"

#ifdef DO_MPI
#include <mpi.h>
#endif

void bml_init(
    int * argc,
    char *** argv);

#ifdef DO_MPI
void bml_initF(
    MPI_Fint* fcomm);
#endif

#endif
