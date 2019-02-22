FROM ubuntu:18.04

ARG QUANDL_API_KEY

RUN apt-get update

RUN apt-get install -y python3-pip libatlas-base-dev python-dev gfortran \
  pkg-config libfreetype6-dev
RUN apt-get install -y tmux zsh

RUN pip3 install zipline tensorflow jupyter matplotlib

RUN QUANDL_API_KEY=$QUANDL_API_KEY zipline ingest -b quandl

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 8888/tcp

CMD ["/bin/bash"]
