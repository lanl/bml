#include "bml.h"

#include <stdio.h>

void bml_log(const bml_log_level_t log_level, const char *message)
{
    printf("%s", message);
}
