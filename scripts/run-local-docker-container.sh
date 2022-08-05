#!/bin/bash

: ${IMAGE:=nicolasbock/bml:latest}

docker pull ${IMAGE} || echo "cannot pull image ${IMAGE}"
docker run --rm \
    $( (( $# == 0 )) && echo "--interactive --tty") \
    --volume "${PWD}":/bml \
    --workdir /bml \
    --user $(id --user):$(id --group) \
    ${IMAGE} "$@"
