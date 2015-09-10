#ifndef __BML_LOGGER_H
#define __BML_LOGGER_H

typedef enum bml_log_level_t {
    BML_LOG_DEBUG,
    BML_LOG_INFO,
    BML_LOG_ERROR
} bml_log_level_t;

void bml_log(const bml_log_level_t log_level, const char *format, ...);

#endif
