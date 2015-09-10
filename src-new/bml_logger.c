#include "bml_logger.h"

#include <stdarg.h>
#include <stdio.h>

void bml_log(const bml_log_level_t log_level, const char *format, ...)
{
    va_list ap;

    va_start(ap, format);
    vprintf(format, ap);
    va_end(ap);
}
