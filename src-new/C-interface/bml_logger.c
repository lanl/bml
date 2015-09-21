#include "bml_logger.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

//static bml_log_level_t global_log_level = BML_LOG_INFO;
static bml_log_level_t global_log_level = BML_LOG_DEBUG;

/** Log a message.
 *
 * \param log_level The log level.
 * \param format The format (as in printf()).
 * \param ap The variadic argument list.
 */
void
bml_log_real (const bml_log_level_t log_level, const char *format, va_list ap)
{
    char new_format[10000];

    if (log_level >= global_log_level)
    {
        if (log_level == BML_LOG_DEBUG)
        {
            snprintf (new_format, 10000, "[DEBUG] %s", format);
        }
        else
        {
            strncpy (new_format, format, 10000);
        }
        vprintf (new_format, ap);
    }
}

/** Log a message.
 *
 * \param log_level The log level.
 * \param format The format (as in printf()).
 */
void
bml_log (const bml_log_level_t log_level, const char *format, ...)
{
    va_list ap;

    va_start (ap, format);
    bml_log_real (log_level, format, ap);
    va_end (ap);
}

/** Log a message with location, i.e. filename and linenumber..
 *
 * \param log_level The log level.
 * \param filename The filename to log.
 * \param linenumber The linenumber.
 * \param format The format (as in printf()).
 */
void
bml_log_location (const bml_log_level_t log_level,
                  const char *filename,
                  const int linenumber, const char *format, ...)
{
    va_list ap;
    char new_format[10000];

    snprintf (new_format, 10000, "[%s:%d] %s", filename, linenumber, format);
    va_start (ap, format);
    bml_log_real (log_level, new_format, ap);
    va_end (ap);
}
