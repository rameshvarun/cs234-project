FROM ubuntu:18.04

ARG QUANDL_API_KEY

RUN apt-get update && apt-get install -y python3-pip libatlas-base-dev \
  python-dev gfortran pkg-config libfreetype6-dev tmux zsh git

RUN pip3 install zipline tensorflow jupyter matplotlib quandl psutil gym black

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN QUANDL_API_KEY=$QUANDL_API_KEY zipline ingest -b quandl

# Install OpenAI baselines.
RUN git clone https://github.com/openai/baselines.git
RUN cd baselines && pip3 install -e .

EXPOSE 8888/tcp

CMD ["/bin/bash"]
