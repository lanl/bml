#!/bin/bash

: ${IMAGE:=nicolasbock/bml:master}

docker pull ${IMAGE}
docker run --interactive --tty --rm \
  --volume ${PWD}:/bml --workdir /bml \
  --user $(id --user):$(id --group) \
  ${IMAGE}
