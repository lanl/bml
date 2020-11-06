#!/bin/bash

set -u -e -x

if [[ -v CI ]]; then
  bundle exec danger
fi
BML_OPENMP=no VERBOSE_MAKEFILE=yes EMACS=emacs26 ./build.sh check_indent
