#!/bin/bash

set -e -u -x

SUDO=$(which sudo || true)

update() {
    for i in $(seq 5); do
        echo "Attempt ${i} to update apt"
        ${SUDO} apt-get update && return
    done
}

install_base() {
    update
    ${SUDO} apt-get install --assume-yes --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget
}

configure_llvm() {
    cat <<EOF | ${SUDO} tee /etc/apt/sources.list.d/clang.list
    # i386 not available
    deb http://apt.llvm.org/noble/ llvm-toolchain-noble main
    # deb-src http://apt.llvm.org/noble/ llvm-toolchain-noble main
    # 17
    deb http://apt.llvm.org/noble/ llvm-toolchain-noble-17 main
    # deb-src http://apt.llvm.org/noble/ llvm-toolchain-noble-17 main
    # 18
    deb http://apt.llvm.org/noble/ llvm-toolchain-noble-18 main
    # deb-src http://apt.llvm.org/noble/ llvm-toolchain-noble-18 main
    # 19
    deb http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main
    # deb-src http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main
    # 20
    deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main
    # deb-src http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main
EOF
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | ${SUDO} tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
    update
}

configure_gcc() {
    if ! apt-cache policy | grep --quiet toolchain; then
        cat <<EOF | ${SUDO} tee /etc/apt/sources.list.d/toolchain.list
        deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu noble main
        # deb-src http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu noble main
EOF
        gpg --keyserver keyserver.ubuntu.com \
            --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F
        gpg --armor --export 60C317803A41BA51845E371A1E9377A2BA9EF27F | ${SUDO} tee /etc/apt/trusted.gpg.d/toolchain.asc
        update
    fi
}

set_timezone() {
    ${SUDO} ln -fs /usr/share/zoneinfo/UTC /etc/localtime
    ${SUDO} apt-get install --assume-yes tzdata
    DEBIAN_FRONTEND=noninteractive ${SUDO} dpkg-reconfigure \
        --frontend noninteractive tzdata
}

install_base_packages() {
    ${SUDO} apt-get install --assume-yes --no-install-recommends \
        apt-utils \
        build-essential \
        bundler \
        cmake cmake-data \
        doxygen \
        emacs \
        gfortran \
        git-core \
        indent \
        less \
        libblas-dev liblapack-dev \
        libopenblas-dev \
        libscalapack-openmpi-dev \
        mpi-default-dev \
        openmpi-bin \
        openssh-client \
        python3-pip python3-wheel python3-pkg-resources \
        sudo \
        texlive \
        texlive-latex-extra \
        texlive-plain-generic \
        valgrind \
        vim
}

install_compilers() {
    ${SUDO} apt-get remove --purge 'libomp*' || echo "no previous clang installation found"

    dpkg -l 'libomp*' || true

    ${SUDO} apt-get install --assume-yes --no-install-recommends \
        clang-19 llvm-19-dev libomp-19-dev libclang-rt-19-dev

    ${SUDO} apt-get install --assume-yes --no-install-recommends \
        gcc-9 g++-9 gfortran-9 \
        gcc-10 g++-10 gfortran-10 \
        gcc-14 g++-14 gfortran-14
}

install_linters() {
    ${SUDO} apt-get install --assume-yes --no-install-recommends \
        python3-bashate
}

if (( $# == 0 )); then
    install_base
    configure_llvm
    configure_gcc
    set_timezone
    install_base_packages
    install_compilers
    install_linters
else
    for command in "$@"; do
        ${command}
    done
fi
