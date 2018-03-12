#!/bin/bash

INDENT_ARGS="-gnu -nut -i4 -bli0 -cli4 -ppi0 -cbi0 -npcs -bfda"

declare -a C_FILES
declare -a FORTRAN_FILES

if [[ $# -gt 0 ]]; then
    for f in "$@"; do
        case "${f##*.}" in
            "c" | "h")
                C_FILES[${#C_FILES[@]}]="$f"
                ;;
            "F90")
                FORTRAN_FILES[${#FORTRAN_FILES[@]}]="$f"
                ;;
            *)
                echo "unknown suffix"
                ;;
        esac
    done
else
    BASEDIR="$(dirname $0)"
    C_FILES=($(find "${BASEDIR}" -name '*.c' -o -name '*.h'))
    FORTRAN_FILES=($(find "${BASEDIR}" -name '*.F90'))
fi

for f in ${C_FILES[@]} ${FORTRAN_FILES[@]}; do
    sed -i -e 's:\s\+$::' "${f}"
done

for f in ${C_FILES[@]}; do
    indent ${INDENT_ARGS} "${f}"
done
