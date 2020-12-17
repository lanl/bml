#!/bin/bash

set -u -e

declare -a actions=(
  "lint"
  "doc"
  "build"
  "interactive"
)

debug=0
default_action="interactive"

while (( $# > 0 )); do
  case $1 in
    -h|--help)
      cat <<EOF
Usage:

-h | --help     This help
-d | --debug    Debug output
ACTION          Known actions: {${actions[@]}}
EOF
      ;;
    -d|--debug)
      debug=1
      ;;
    lint)
      actions=( ${actions[@]} $1)
      ;;
    interactive)
      bash -l -i
      ;;
    *)
      echo "unknown action $1"
      exit 1
      ;;
  esac
  shift
done

if (( debug == 1 )); then
  PS4=""
  set -x
fi

      if [[ -v CI ]]; then
        bundle exec danger
      fi
      BML_OPENMP=no VERBOSE_MAKEFILE=yes EMACS=emacs26 ./build.sh check_indent
case ${
