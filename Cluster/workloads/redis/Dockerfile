FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    openssh-server \
    python3-pip \
    vim \
    redis-server

RUN pip3 install aiohttp cchardet

RUN mkdir /work
COPY . /work/.
WORKDIR /work

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["--help"]
