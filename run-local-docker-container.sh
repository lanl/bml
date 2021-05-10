#!/bin/bash

: ${IMAGE:=nicolasbock/bml:latest}

docker pull ${IMAGE}
docker run --interactive --tty --rm \
  --volume ${PWD}:/bml --workdir /bml \
  --user $(id --user):$(id --group) \
  ${IMAGE}
