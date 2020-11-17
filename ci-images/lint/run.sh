#!/bin/bash

set -u -e -x

basedir=$(readlink --canonicalize $(dirname $0)/../..)
dockerfile=${basedir}/Dockerfile

docker build --no-cache --file ${dockerfile} ${basedir}
ID=$(docker build --quiet --file ${dockerfile} ${basedir})
docker run \
  --workdir /github/workspace \
  -v "/var/run/docker.sock":"/var/run/docker.sock" \
  -v "${basedir}":"/github/workspace" \
  ${ID}
