#!/bin/bash
set -e

if [[ -z $1 ]]; then
    echo "NO TARGET HOST NODE"
    exit
else
    TARGET_HOST=$1
    TARGET_HOST_NAME=$(ssh $1 hostname)
fi

if [[ -z $2 ]]; then
    WORKLOAD_ARRAY=(0 1 2 3 4 5 6 7)
else
    WORKLOAD_ARRAY=($2 $3 $4 $5 $6 $7 $8 $9)
fi

CPUS='2'
MEMORY='8G'

IFS=$'\n' WORKLOAD_ARRAY=($(sort <<<"${WORKLOAD_ARRAY[*]}")); unset IFS
NUM_REDIS_1=$(grep -o '1' <<< ${WORKLOAD_ARRAY[*]} | wc -l)

INDX=0
REDIS_1_PORT=6381 # 6381, 6382, ..., 6389, 6390, 6391, ...
for i in "${WORKLOAD_ARRAY[@]}"; do
    PORT=$(($INDX+8081))
    echo "$i"
    case "$i" in
        1) DOCKER_IMAGE=aiohttp/redis:latest
        echo "${PORT}: ${DOCKER_IMAGE} redis-1 ($REDIS_1_PORT)"
        docker service create -p ${PORT}:8080 -p ${REDIS_1_PORT}:6379 --limit-cpu ${CPUS} --limit-memory ${MEMORY} --constraint "node.hostname==${TARGET_HOST_NAME}" ${DOCKER_IMAGE} redis 2
        REDIS_1_PORT=$(($REDIS_1_PORT+1))
        ;;

        2) DOCKER_IMAGE=aiohttp/mxnet-model-server:latest
        echo "${PORT}: ${DOCKER_IMAGE} mms-1 (#redis-1:${NUM_REDIS_1})"
        docker service create -p ${PORT}:8079 --limit-cpu ${CPUS} --limit-memory ${MEMORY} --constraint "node.hostname==${TARGET_HOST_NAME}" ${DOCKER_IMAGE} mms 1 ${TARGET_HOST} ${NUM_REDIS_1} 2 1  # resnet-152, batch_size=1
        ;;

        3) DOCKER_IMAGE=aiohttp/checksum:latest
        echo "${PORT}: ${DOCKER_IMAGE} checksum"
        docker service create -p ${PORT}:8080 --limit-cpu ${CPUS} --limit-memory ${MEMORY} --constraint "node.hostname==${TARGET_HOST_NAME}" -e SELF_HOST=${TARGET_HOST} -e NUM_REDIS=${NUM_REDIS_1} -e SELF_PORT=${PORT} ${DOCKER_IMAGE}
        ;;

        4) DOCKER_IMAGE=aiohttp/ycsb-memcached:latest
        echo "${PORT}: ${DOCKER_IMAGE} ycsb-memcached-A"
        docker service create -p ${PORT}:8080 --limit-cpu ${CPUS} --limit-memory ${MEMORY} --constraint "node.hostname==${TARGET_HOST_NAME}" -e SELF_HOST=${TARGET_HOST} ${DOCKER_IMAGE} workloada
        ;;

        5) DOCKER_IMAGE=aiohttp/ycsb-memcached:latest
        echo "${PORT}: ${DOCKER_IMAGE} ycsb-memcached-B"
        docker service create -p ${PORT}:8080 --limit-cpu ${CPUS} --limit-memory ${MEMORY} --constraint "node.hostname==${TARGET_HOST_NAME}" -e SELF_HOST=${TARGET_HOST} ${DOCKER_IMAGE} workloadb
        ;;

        6) DOCKER_IMAGE=aiohttp/scenedetect:latest
        echo "${PORT}: ${DOCKER_IMAGE} scenedetect"
        docker service create -p ${PORT}:8080 --limit-cpu ${CPUS} --limit-memory ${MEMORY} --constraint "node.hostname==${TARGET_HOST_NAME}" -e SELF_HOST=${TARGET_HOST} -e SELF_PORT=${PORT} ${DOCKER_IMAGE}
        ;;  

        7) DOCKER_IMAGE=aiohttp/isr:latest
        echo "${PORT}: ${DOCKER_IMAGE} isr"
        docker service create -p ${PORT}:8080 --limit-cpu ${CPUS} --limit-memory ${MEMORY} --constraint "node.hostname==${TARGET_HOST_NAME}" -e SELF_HOST=${TARGET_HOST} -e SELF_PORT=${PORT} -e NUM_REDIS=${NUM_REDIS_1} ${DOCKER_IMAGE}
        ;;

        0)
        echo "NO WORKLOAD"
        ;;
    esac
    INDX=$(($INDX+1))
done
echo ${WORKLOAD_ARRAY[@]}

