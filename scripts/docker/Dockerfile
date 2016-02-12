FROM quay.io/travisci/travis-ruby
MAINTAINER nicolasbock@gmail.com
WORKDIR /
RUN \
      apt-get update; \
      apt-get -y install python-software-properties; \
      apt-add-repository -y ppa:ubuntu-toolchain-r/test; \
      apt-add-repository -y ppa:george-edison55/precise-backports; \
      apt-get update; \
      apt-get -y install cmake cmake-data \
      gcc${VER} g++${VER} gfortran${VER} \
      git \
      libblas3gf liblapack3gf \
      python-pip valgrind; \
      pip install codecov
CMD /bin/bash
