FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    vim \
    git \
    redis-server

RUN pip3 install redis requests aiohttp numpy ISR

COPY docker-entrypoint.sh /
COPY aiohttp-isr.py /
RUN chmod +x /docker-entrypoint.sh
RUN chmod +x /aiohttp-isr.py

CMD [ "/docker-entrypoint.sh" ]