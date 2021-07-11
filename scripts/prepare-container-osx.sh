#!/bin/bash

set -e -u -x

SUDO=$(which sudo || true)

for i in $(seq 5); do
  brew update && break
done
brew upgrade
brew search
