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
    deb http://apt.llvm.org/focal/ llvm-toolchain-focal main
    # deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal main
    # 11
    deb http://apt.llvm.org/focal/ llvm-toolchain-focal-11 main
    # deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-11 main
    # 12
    deb http://apt.llvm.org/focal/ llvm-toolchain-focal-12 main
    # deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-12 main
    # 13
    deb http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main
    # deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main
    # 14
    deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main
    # deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main
    # 15
    deb http://apt.llvm.org/focal/ llvm-toolchain-focal-15 main
    # deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-15 main
    # 16
    deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main
    # deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main
EOF
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | ${SUDO} tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
    update
}

configure_gcc() {
    if ! apt-cache policy | grep --quiet toolchain; then
        cat <<EOF | ${SUDO} tee /etc/apt/sources.list.d/toolchain.list
        deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu focal main
        # deb-src http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu focal main
EOF
        gpg --keyserver keyserver.ubuntu.com \
            --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F
        gpg --armor --export 60C317803A41BA51845E371A1E9377A2BA9EF27F | ${SUDO} tee /etc/apt/trusted.gpg.d/toolchain.asc
        update
    fi
}

configure_emacs() {
    cat <<EOF | ${SUDO} tee /etc/apt/sources.list.d/emacs.list
    deb http://ppa.launchpad.net/kelleyk/emacs/ubuntu focal main
    # deb-src http://ppa.launchpad.net/kelleyk/emacs/ubuntu focal main
EOF
    gpg --keyserver keyserver.ubuntu.com \
        --recv-keys 873503A090750CDAEB0754D93FF0E01EEAAFC9CD
    gpg --armor --export 873503A090750CDAEB0754D93FF0E01EEAAFC9CD | ${SUDO} tee /etc/apt/trusted.gpg.d/emacs.asc
    update
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
        gfortran \
        bundler \
        cmake cmake-data \
        emacs27 \
        git-core \
        indent \
        less \
        libblas-dev liblapack-dev \
        libscalapack-openmpi-dev \
        libopenblas-dev \
        mpi-default-dev \
        openmpi-bin \
        openssh-client \
        python python3-pip python3-wheel python3-pkg-resources \
        sudo \
        texlive \
        valgrind \
        vim
}

install_compilers() {
    ${SUDO} apt-get remove --purge 'libomp*' || echo "no previous clang installation found"

    dpkg -l 'libomp*' || true

    ${SUDO} apt-get install --assume-yes --no-install-recommends \
        clang-16 llvm-16-dev libomp-16-dev libclang-rt-16-dev

    ${SUDO} apt-get install --assume-yes --no-install-recommends \
        gcc-9 g++-9 gfortran-9 \
        gcc-10 g++-10 gfortran-10
}

install_linters() {
    ${SUDO} pip install --system bashate
}

if (( $# == 0 )); then
    install_base
    configure_llvm
    configure_gcc
    configure_emacs
    set_timezone
    install_base_packages
    install_compilers
    install_linters
else
    for command in "$@"; do
        ${command}
    done
fi
