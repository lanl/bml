#!/bin/bash

set -u -e

clean_build=0

declare -a ci_args=()

while (( $# > 0 )); do
  case $1 in
    -h|--help)
      cat <<EOF
Usage:

$(basename $0) [OPTIONS] {lint,build,doc,interactive}

Options:

-h | --help          This help
-c | --clean-build   Rebuild the docker image (default
                     is to use a cached image if one
                     exists)
EOF
      ;;
    -c|--clean-build)
      clean_build=1
      ;;
    *)
      ci_args=( ${ci_args[@]} $1)
      ;;
  esac
  shift
done

set -x

basedir=$(readlink --canonicalize $(dirname $0))
dockerfile=${basedir}/Dockerfile

docker build \
  $((( clean_build == 1 )) && echo "--no-cache") \
  --file ${dockerfile} \
  --tag bml-ci-image \
  ${basedir}
ID=$(docker build --quiet --file ${dockerfile} ${basedir})

docker run \
  --workdir /github/workspace \
  --interactive \
  --tty \
  -v "/var/run/docker.sock":"/var/run/docker.sock" \
  -v "${basedir}":"/github/workspace" \
  ${ID} \
  ${ci_args[@]}
