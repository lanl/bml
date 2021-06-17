FROM ubuntu:bionic

COPY scripts/prepare-container.sh /usr/sbin
RUN /usr/sbin/prepare-container.sh

WORkDIR /root
