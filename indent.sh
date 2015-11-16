#!/bin/bash

INDENT_ARGS="-gnu -nut -i4 -bli0 -cli4 -ppi0 -cbi0 -npcs -bfda"

if [[ $# -gt 0 ]]; then
    indent ${INDENT_ARGS} $*
else
    BASEDIR="$(dirname $0)"
    indent ${INDENT_ARGS} \
           "${BASEDIR}"/src/C-interface/{,dense,ellpack}/*.{c,h} \
           "${BASEDIR}"/tests/*.{c,h}
fi
