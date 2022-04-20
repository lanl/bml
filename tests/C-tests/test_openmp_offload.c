#include <omp.h>

int
main(
    int argc,
    char **argv)
{
    int is_device = 1;

#pragma omp target map(from : is_device)
    {
        is_device = omp_is_initial_device();
    }

    // returns 0 if executed on device, 1 otherwise
    return is_device;
}
