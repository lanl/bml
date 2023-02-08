
#ifndef ROCSPARSE_EXPORT_H
#define ROCSPARSE_EXPORT_H

#ifdef ROCSPARSE_STATIC_DEFINE
#  define ROCSPARSE_EXPORT
#  define ROCSPARSE_NO_EXPORT
#else
#  ifndef ROCSPARSE_EXPORT
#    ifdef rocsparse_EXPORTS
        /* We are building this library */
#      define ROCSPARSE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define ROCSPARSE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef ROCSPARSE_NO_EXPORT
#    define ROCSPARSE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef ROCSPARSE_DEPRECATED
#  define ROCSPARSE_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef ROCSPARSE_DEPRECATED_EXPORT
#  define ROCSPARSE_DEPRECATED_EXPORT ROCSPARSE_EXPORT ROCSPARSE_DEPRECATED
#endif

#ifndef ROCSPARSE_DEPRECATED_NO_EXPORT
#  define ROCSPARSE_DEPRECATED_NO_EXPORT ROCSPARSE_NO_EXPORT ROCSPARSE_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef ROCSPARSE_NO_DEPRECATED
#    define ROCSPARSE_NO_DEPRECATED
#  endif
#endif

#endif /* ROCSPARSE_EXPORT_H */
