#include "magma_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

int
main(
    int argc,
    char *argv[])
{
    /* matrix dimensions */
    int na = 100;

    printf("==========================================\n");
    printf("MAGMA simple test\n");

    printf("Matrix size: %d\n", na);

    /* allocate the matrices needed */
    double *da;
    magma_int_t ret = magma_dmalloc(&da, na * na);
    if (ret != MAGMA_SUCCESS)
    {
        printf("MAGMA allocation failed!\n");
        return 1;
    }

    printf("Set matrix on CPU...\n");
    double *ha = calloc(na * na, sizeof(double));
    for (int i = 0; i < na; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            ha[i * na + j] = rand() / (double) RAND_MAX;
            if (i != j)
                ha[j * na + i] = ha[i * na + j];
        }
    }

    int device;
    magma_getdevice(&device);
    magma_queue_t queue;
    magma_queue_create(device, &queue);

    printf("Set matrix on GPU...\n");
    magma_dsetmatrix(na, na, ha, na, da, na, queue);

    magma_queue_sync(queue);

    /* copy data back to CPU */
    double *hb = calloc(na * na, sizeof(double));
    magma_dgetmatrix(na, na, da, na, hb, na, queue);

    /* check data on CPU */
    int status = 0;
    for (int i = 0; i < na * na; i++)
    {
        if (fabs(ha[i] - hb[i]) > 1.e-6)
        {
            status = 1;
            printf("index %d, ha = %le, hb = %le\n", i, ha[i], hb[i]);
        }
    }
    if (status == 1)
        return 1;

    printf("Free resources...\n");
    ret = magma_free(da);
    if (ret != MAGMA_SUCCESS)
    {
        printf("MAGMA free failed!\n");
        return 1;
    }

    free(ha);
    free(hb);

    return 0;
}
