FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
    openjdk-8-jre-headless \
    python \
    vim \
    git \
    tar \
    iputils-ping \
    maven \
    default-jdk \
    netcat

WORKDIR /
RUN git clone http://github.com/brianfrankcooper/YCSB.git
RUN cd YCSB
RUN mvn -pl com.yahoo.ycsb:memcached-binding -am clean package

COPY memcached.conf /etc/memcached.conf

RUN pip3 install aiohttp cchardet

COPY *.py /YCSB/
COPY *.sh /YCSB/

WORKDIR /YCSB
ENTRYPOINT ["/YCSB/docker-entrypoint.sh"]
CMD ["--help"]

