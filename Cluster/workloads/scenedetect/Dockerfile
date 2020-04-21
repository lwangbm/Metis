FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    vim \
    redis-server

RUN pip3 install opencv-python scenedetect aiohttp
RUN apt-get install -y libxext6 libxrender-dev
RUN pip3 install redis requests python-memcached
RUN apt-get install -y libsm6

COPY hiking.mp4 /
COPY docker-entrypoint.sh /
COPY aiohttp-scenedetect.py /
RUN chmod +x /docker-entrypoint.sh
RUN chmod +x /aiohttp-scenedetect.py

CMD [ "/docker-entrypoint.sh" ]