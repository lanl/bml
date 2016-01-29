#!/bin/bash

set -x

VER=${GCC_VERSION:+-${GCC_VERSION}}

apt-get update
apt-get -y install python-software-properties
apt-add-repository -y ppa:ubuntu-toolchain-r/test
apt-add-repository -y ppa:george-edison55/precise-backports
apt-get update
apt-get -y install cmake cmake-data \
  gcc${VER} g++${VER} gfortran${VER} \
  git \
  libblas3gf liblapack3gf \
  python-pip valgrind
pip install codecov
git clone https://github.com/qmmd/bml.git
cd bml
git checkout -b develop origin/develop

export CC=gcc${VER}
export CXX=g++${VER}
export FC=gfortran${VER}
export VERBOSE_MAKEFILE=yes

./build.sh testing

/bin/bash
