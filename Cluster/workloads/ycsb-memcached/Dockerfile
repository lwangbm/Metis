FROM maven:3.6.3-jdk-8

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    vim \
    git \
    tar \
    netcat \
    iputils-ping \
    memcached

WORKDIR /
RUN git clone https://github.com/brianfrankcooper/YCSB.git
WORKDIR /YCSB
RUN mvn -pl site.ycsb:memcached-binding -am clean package

COPY memcached.conf /etc/memcached.conf

RUN pip3 install aiohttp cchardet

COPY *.py /YCSB/
COPY *.sh /YCSB/

WORKDIR /YCSB
ENTRYPOINT ["/YCSB/docker-entrypoint.sh"]
CMD ["--help"]

