#include "bml_logger.h"

#include <stdarg.h>
#include <stdio.h>

static bml_log_level_t global_log_level = BML_LOG_INFO;

void bml_log(const bml_log_level_t log_level, const char *format, ...)
{
    va_list ap;

    if(log_level >= global_log_level) {
        va_start(ap, format);
        vprintf(format, ap);
        va_end(ap);
    }
}
