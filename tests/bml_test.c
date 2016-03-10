#define _POSIX_C_SOURCE 200809L

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "bml_test.h"

const int NUM_TESTS = 13;

const char *test_name[] =
    { "add", "allocate", "convert", "copy", "diagonalize", "multiply",
    "norm", "normalize", "scale", "submatrix", "threshold", "trace", "transpose"
};

const char *test_description[] = {
    "Add two bml matrices",
    "Allcate bml matrices",
    "Convert bml matrices",
    "Copy bml matrices",
    "Diagonalize a bml matrix",
    "Multiply two bml matrices",
    "Norm of bml matrix",
    "Normalize bml matrices",
    "Scale bml matrices",
    "Submatrix bml matrices",
    "Threshold bml matrices",
    "Trace of bml matrices",
    "Transpose of bml matrices"
};

const test_function_t testers[] = {
    test_add,
    test_allocate,
    test_convert,
    test_copy,
    test_diagonalize,
    test_multiply,
    test_norm,
    test_normalize,
    test_scale,
    test_submatrix,
    test_threshold,
    test_trace,
    test_transpose
};

void
print_usage(
    void)
{
    printf("Usage:\n");
    printf("\n");
    printf("-h | --help           This help\n");
    printf("-n | --testname TEST  Run test TEST\n");
    printf("-t | --type T         Choose the matrix type\n");
    printf("-p | --precision P    Choose matrix precision\n");
    printf("-l | --list           List all available tests\n");
    printf("-N | --N N            Test N x N matrices\n");
    printf("-M | --M M            Pass an extra parameter M to the test\n");
    printf("\n");
    printf("Recognized types:\n");
    printf("\n");
    printf("  dense\n");
    printf("  ellpack\n");
    printf("\n");
    printf("Recognized precisions:\n");
    printf("\n");
    printf("  single_real\n");
    printf("  double_real,\n");
    printf("  single_complex\n");
    printf("  double_complex\n");
    printf("\n");

    int max_width = 0;
    for (int i = 0; i < NUM_TESTS; i++)
    {
        if (strlen(test_name[i]) > max_width)
        {
            max_width = strlen(test_name[i]);
        }
    }
    char desc_format[100];
    snprintf(desc_format, 100, "%%%ds    %%s\n", max_width);

    printf("Available tests:\n");
    printf("\n");
    for (int i = 0; i < NUM_TESTS; i++)
    {
        printf(desc_format, test_name[i], test_description[i]);
    }
}

int
main(
    int argc,
    char **argv)
{
    int N = 11;
    int M = -1;
    char *test = NULL;
    int test_index = -1;
    bml_matrix_type_t matrix_type = dense;
    bml_matrix_precision_t precision = single_real;

    const char *short_options = "hn:t:p:lN:M:";
    const struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"testname", required_argument, NULL, 'n'},
        {"type", required_argument, NULL, 't'},
        {"precision", required_argument, NULL, 'p'},
        {"list", no_argument, NULL, 'l'},
        {"N", required_argument, NULL, 'N'},
        {"M", required_argument, NULL, 'M'},
        {NULL, 0, NULL, 0}
    };
    int c;

    while ((c =
            getopt_long(argc, argv, short_options, long_options, NULL)) != -1)
    {
        switch (c)
        {
            case 'h':
                print_usage();
                return 0;
                break;
            case 'n':
            {
                if (test)
                {
                    free(test);
                }
                test = strdup(optarg);
                for (test_index = 0; test_index < NUM_TESTS; test_index++)
                {
                    if (strcasecmp(test, test_name[test_index]) == 0)
                    {
                        break;
                    }
                }
                if (test_index == NUM_TESTS)
                {
                    fprintf(stderr, "unknown test %s\n", test);
                    return 1;
                }
                break;
            }
            case 't':
                if (strcasecmp(optarg, "dense") == 0)
                {
                    matrix_type = dense;
                }
                else if (strcasecmp(optarg, "ellpack") == 0)
                {
                    matrix_type = ellpack;
                }
                else
                {
                    fprintf(stderr, "unknown matrix type %s\n", optarg);
                }
                break;
            case 'p':
                if (strcasecmp(optarg, "single_real") == 0)
                {
                    precision = single_real;
                }
                else if (strcasecmp(optarg, "double_real") == 0)
                {
                    precision = double_real;
                }
                else if (strcasecmp(optarg, "single_complex") == 0)
                {
                    precision = single_complex;
                }
                else if (strcasecmp(optarg, "double_complex") == 0)
                {
                    precision = double_complex;
                }
                else
                {
                    fprintf(stderr, "unknow matrix precision %s\n", optarg);
                }
                break;
            case 'l':
                for (int i = 0; i < NUM_TESTS; i++)
                {
                    printf("%s", test_name[i]);
                    if (i < NUM_TESTS - 1)
                    {
                        printf(", ");
                    }
                }
                printf("\n");
                break;
            case 'N':
                N = strtol(optarg, NULL, 10);
                break;
            case 'M':
                M = strtol(optarg, NULL, 10);
                break;
            default:
                fprintf(stderr, "unknown option\n");
                return 1;
                break;
        }
    }

    if (!test)
    {
        fprintf(stderr, "missing test\n");
        return 1;
    }

    if (M < 0)
    {
        M = N;
    }

    fprintf(stderr, "%s\n", test);
    fprintf(stderr, "N = %d\n", N);
    free(test);
    return testers[test_index] (N, matrix_type, precision, M);
}
