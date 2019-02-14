#!/bin/bash

set -x

basedir=$(readlink --canonicalize $(dirname $0))
files=$(git ls-tree --full-tree -r HEAD \
    | grep '.\(c\|h\|F90\)$' \
    | awk '{print $4}')
pushd "${basedir}" || exit
ctags --recurse --C-kinds=+lxzLp --Fortran-kinds=+LP ${files}
etags ${files}
popd || exit
