#!/bin/bash

set -x -e

: ${IMAGE_TAG:=bml}
: ${IMAGE_VERSION:=test}
: ${PUSH_IMAGE:=no}

docker build --pull --tag nicolasbock/${IMAGE_TAG}:${IMAGE_VERSION} .
if [[ ${PUSH_IMAGE} = yes ]]; then
  docker push nicolasbock/${IMAGE_TAG}:${IMAGE_VERSION}
fi
