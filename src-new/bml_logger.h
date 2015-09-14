#ifndef __BML_LOGGER_H
#define __BML_LOGGER_H

#include "bml_types.h"

typedef enum {
    BML_LOG_DEBUG,
    BML_LOG_INFO,
    BML_LOG_WARNING,
    BML_LOG_ERROR
} bml_log_level_t;

#define LOG_DEBUG(format, ...) bml_log_location(BML_LOG_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) bml_log(BML_LOG_INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) bml_log_location(BML_LOG_WARNING, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) bml_log_location(BML_LOG_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)

void bml_log(const bml_log_level_t log_level, const char *format, ...);

void bml_log_location(const bml_log_level_t log_level,
                      const char *filename,
                      const int linenumber,
                      const char *format, ...);

#endif
