FROM ubuntu:bionic

COPY prepare-container.sh /usr/sbin
RUN /usr/sbin/prepare-container.sh

WORkDIR /root
