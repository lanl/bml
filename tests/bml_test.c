#define _POSIX_C_SOURCE 200809L

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

const int NUM_TESTS = 2;
const char *test_name[] = { "new", "multiply" };

const char *test_description[] = {
    "Instantiate a new bml matrix",
    "Multiply two bml matrices"
};

void
print_usage(
    void)
{
    printf("Usage:\n");
    printf("\n");
    printf("-h | --help       This help\n");
    printf("-t | --test TEST  Run test TEST\n");
    printf("-l | --list       List all available tests\n");
    printf("-N | --N N        Test N x N matrices\n");
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
    snprintf(desc_format, 100, "%%%ds   %%s\n", max_width);

    printf("Available tests:\n");
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

    const char *short_options = "ht:lN:";
    const struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"test", required_argument, NULL, 't'},
        {"list", no_argument, NULL, 'l'},
        {"N", required_argument, NULL, 'N'},
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
            case 't':
            {
                char *test_str = strdup(optarg);
                int i;
                for (i = 0; i < NUM_TESTS; i++)
                {
                    if (strcasecmp(test_str, test_name[i]) == 0)
                    {
                        break;
                    }
                }
                if (i == NUM_TESTS)
                {
                    fprintf(stderr, "unknown test %s\n", test_str);
                    return 1;
                }
                break;
            }
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
            default:
                fprintf(stderr, "unknown option\n");
                return 1;
                break;
        }
    }

    fprintf(stderr, "N = %d\n", N);

    return 0;
}
