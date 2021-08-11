#!/bin/bash

set -e -u -x

basedir=$(readlink --canonicalize $(dirname $0)/..)
pushd "${basedir}"
readarray -t files < <(git ls-files '*.c' '*.h' '*.F90')
ctags "${files[@]}"
etags "${files[@]}"
popd
