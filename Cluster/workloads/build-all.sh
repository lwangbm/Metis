#!/bin/bash
for folder in redis mxnet-model-server isr checksum ycsb-memcached scenedetect; do
    cd $folder
    docker build -t "aiohttp/${folder}" .
    cd ..
done

cd nginx
docker build -t "nginx" .
cd ..
