FROM ubuntu:bionic
MAINTAINER nicolasbock@gmail.com

COPY scripts/prepare-container.sh /usr/sbin
RUN /usr/sbin/prepare-container.sh

WORkDIR /root
