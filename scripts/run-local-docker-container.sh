#!/bin/bash

: ${IMAGE:=nicolasbock/bml:latest}

docker pull "${IMAGE}" || echo "cannot pull image ${IMAGE}"
docker run --rm \
    $( (( $# == 0 )) && echo "--interactive --tty") \
    --volume "${PWD}":/bml \
    --workdir /bml \
    ${IMAGE} sudo --user ubuntu $( (( $# == 0 )) && echo "bash --login" ) "$@"
