FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    vim \
    redis-server

RUN pip3 install redis requests python-memcached aiohttp

COPY docker-entrypoint.sh /
COPY aiohttp-checksum.py /
RUN chmod +x /docker-entrypoint.sh
RUN chmod +x /aiohttp-checksum.py

CMD ["/docker-entrypoint.sh"]
