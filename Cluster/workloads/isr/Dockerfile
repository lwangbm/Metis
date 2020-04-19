FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    vim \
    redis-server

RUN pip3 install redis requests

COPY docker-entrypoint.sh /
COPY aiohttp-isr.py /
RUN chmod +x /docker-entrypoint.sh
RUN chmod +x /aiohttp-isr.py

CMD [ "/docker-entrypoint.sh" ]