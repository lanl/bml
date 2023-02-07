FROM ubuntu:bionic
LABEL org.opencontainers.image.authors="nicolasbock@gmail.com"

COPY scripts/prepare-container.sh /usr/sbin
RUN /usr/sbin/prepare-container.sh

WORKDIR /root
