#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

long int get_reduction_old(const int max_n)
{
    long int emax = 0;

    printf("using < OpenMP-3.1\n");

    /* Start a parallel section. All threads are started.
     */
#pragma omp parallel
    {
        /* The thread local variable. This is used to find the max within the
         * thread.
         */
        int _max = 0;

#pragma omp master
        printf("running on %d threads\n", omp_get_num_threads());

        /* Run the for loop across the thread pool. The local max is updated
         * appropriately per thread.
         */
#pragma omp for nowait
        for(int i = 0; i < max_n; i++)
        {
            _max = i;
        }

        /* The 'critical' section specifies that only one thread at a time can
         * enter it. We use it to update the global max. The 'nowait' clause
         * in the for loop ensures that we don't block unecessarily already
         * earlier on the for loop.
         */
#pragma omp critical
        if (_max > emax)
        {
            emax = _max;
        }
    }

    return emax;
}

long int get_reduction_new(const int max_n)
{
    long int emax = 0;

    printf("using >= OpenMP-3.1\n");

#pragma omp parallel
    {
#pragma omp master
        printf("running on %d threads\n", omp_get_num_threads());
    }

#pragma omp parallel for reduction(max:emax)
    for(int i = 0; i < max_n; i++)
    {
        emax = i;
    }

    return emax;
}

int main(int argc, char **argv)
{
    short use_max_reduction = 0;
    long int emax;
    long int max_n = 1000;

    const char *shortopt = "nN:";
    char ch;
    while ((ch = getopt(argc, argv, shortopt)) != -1)
    {
        switch(ch)
        {
            case 'n':
                use_max_reduction = 1;
                break;

            case 'N':
                max_n = strtol(optarg, NULL, 10);
                break;

            default:
                printf("error parsing command line\n");
                return -1;
                break;
        }
    }

    if (use_max_reduction)
    {
        emax = get_reduction_new(max_n);
    }
    else
    {
        emax = get_reduction_old(max_n);
    }

    printf("emax = %ld\n", emax);

    if (emax != max_n - 1)
    {
        printf("incorrect max reduction\n");
        return 1;
    }

    return 0;
}
