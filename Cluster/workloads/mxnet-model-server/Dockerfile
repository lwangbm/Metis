FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
    openjdk-8-jre-headless \
    python3-pip \
    vim \
    git \
    tar

RUN apt-get update && apt-get install -y redis-server

RUN pip3 install aiohttp cchardet redis mxnet-mkl mxnet-model-server

RUN mkdir /mxnet-model-server/
COPY image /mxnet-model-server/image
COPY *.py /mxnet-model-server/
COPY *.sh /mxnet-model-server/

WORKDIR /mxnet-model-server/
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["--help"]
