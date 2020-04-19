#!/bin/bash
# usage:
# bash script.sh $NGINX_HOST 0 2 3 4 5 6 7
set -e

LOCAL_PRIVATE_IP=$(hostname -i) # itself

# start memcached
sudo service memcached restart

# (1, 2, 3, 4, 0, 0, 0, 0)
if [[ -z $1 ]]; then
    echo "NO NGINX HOST"
    exit
else
    NGINX_HOST=$1
fi

if [[ -z $2 ]]; then
    WORKLOAD_ARRAY=(0 2 3 4 5 6 7)
else
    WORKLOAD_ARRAY=($2 $3 $4 $5 $6 $7 $8 $9) # since $1 is for NGINX_HOST
fi

CPUSET_ARRAY=("0,8" "1,9" "2,10" "3,11"  "4,12" "5,13" "6,14" "7,15")
CPUS='2'
MEMORY='8G'

IFS=$'\n' WORKLOAD_ARRAY=($(sort <<<"${WORKLOAD_ARRAY[*]}")); unset IFS
NUM_REDIS_1=$(grep -o '1' <<< ${WORKLOAD_ARRAY[*]} | wc -l)

INDX=0
REDIS_1_PORT=6381 # 6381, 6382, ..., 6389, 6390, 6391, ...
# --cpuset-cpus
for i in "${WORKLOAD_ARRAY[@]}"; do
    PORT=$(($INDX+8081))
    echo "$i"
    case "$i" in
        1) DOCKER_IMAGE=aiohttp/redis:latest
        echo "${PORT}: ${DOCKER_IMAGE} redis-1 ($REDIS_1_PORT)"
        CONTAINER_ID=$(docker run --rm -d --name "redis-1-${PORT}-$(date +%H%M%S)" -p ${PORT}:8080 -p ${REDIS_1_PORT}:6379 ${DOCKER_IMAGE} redis 2)
        docker update --memory=${MEMORY} --cpus=${CPUS} ${CONTAINER_ID}
        REDIS_1_PORT=$(($REDIS_1_PORT+1))
        ;;

        2) DOCKER_IMAGE=aiohttp/mxnet-model-server:latest
        echo "${PORT}: ${DOCKER_IMAGE} mms-1 (#redis-1:${NUM_REDIS_1})"
        CONTAINER_ID=$(docker run --rm -d --name "mms-1-${PORT}-$(date +%H%M%S)" -p ${PORT}:8079 ${DOCKER_IMAGE} mms 1 ${LOCAL_PRIVATE_IP} ${NUM_REDIS_1} 2 1)  # resnet-152, batch_size=1
        docker update --memory=${MEMORY} --cpus=${CPUS} ${CONTAINER_ID}
        ;;

        3) DOCKER_IMAGE=aiohttp/checksum:latest
        echo "${PORT}: ${DOCKER_IMAGE} checksum"
        CONTAINER_ID=$(docker run --rm -d --name "checksum-${PORT}-$(date +%H%M%S)" -p ${PORT}:8080 -e SELF_HOST=${LOCAL_PRIVATE_IP} -e NUM_REDIS=${NUM_REDIS_1} -e SELF_PORT=${PORT} ${DOCKER_IMAGE})
        docker update --memory=${MEMORY} --cpus=${CPUS} ${CONTAINER_ID}
        ;;

        4) DOCKER_IMAGE=aiohttp/ycsb-memcached:latest
        echo "${PORT}: ${DOCKER_IMAGE} ycsb-memcached-A"
        CONTAINER_ID=$(docker run --rm -d --name "ycsb-memcached-A-${PORT}-$(date +%H%M%S)" -p ${PORT}:8080 -e SELF_HOST=${LOCAL_PRIVATE_IP} ${DOCKER_IMAGE} workloada)
        docker update --memory=${MEMORY} --cpus=${CPUS} ${CONTAINER_ID}
        ;;

        5) DOCKER_IMAGE=aiohttp/ycsb-memcached:latest
        echo "${PORT}: ${DOCKER_IMAGE} ycsb-memcached-B"
        CONTAINER_ID=$(docker run --rm -d --name "ycsb-memcached-B-${PORT}-$(date +%H%M%S)" -p ${PORT}:8080 -e SELF_HOST=${LOCAL_PRIVATE_IP} ${DOCKER_IMAGE} workloadb)
        docker update --memory=${MEMORY} --cpus=${CPUS} ${CONTAINER_ID}
        ;;

        6) DOCKER_IMAGE=aiohttp/scenedetect:latest
        echo "${PORT}: ${DOCKER_IMAGE} scenedetect"
        CONTAINER_ID=$(docker run --rm -d --name "scenedetect-${PORT}-$(date +%H%M%S)" -p ${PORT}:8080 -e SELF_HOST=${LOCAL_PRIVATE_IP} -e SELF_PORT=${PORT} ${DOCKER_IMAGE})
        docker update --memory=${MEMORY} --cpus=${CPUS} ${CONTAINER_ID}
        ;;  

        7) DOCKER_IMAGE=aiohttp/isr:latest
        echo "${PORT}: ${DOCKER_IMAGE} isr"
        CONTAINER_ID=$(docker run --rm -d --name "isr-${PORT}-$(date +%H%M%S)" -p ${PORT}:8080 -e SELF_HOST=${LOCAL_PRIVATE_IP} -e NUM_REDIS=${NUM_REDIS_1} -e SELF_PORT=${PORT} ${DOCKER_IMAGE})
        docker update --memory=${MEMORY} --cpus=${CPUS} ${CONTAINER_ID}
        ;;

        0)
        echo "NO WORKLOAD"
        ;;
    esac
    INDX=$(($INDX+1))
done
echo ${WORKLOAD_ARRAY[@]}
