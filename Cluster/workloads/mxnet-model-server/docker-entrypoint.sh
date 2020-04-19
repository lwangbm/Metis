#!/bin/bash
# docker run --rm -d -p ${PORT}:8079 ${DOCKER_IMAGE} mms 1 ${LOCAL_PRIVATE_IP} ${NUM_REDIS_1} 3 512
set -e
if [ "$1" = 'mms' ]; then
    if [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
        echo "mms <1,2> <redis_host> <#redis> <1,2,3> <batch_size>"
        sleep infinity # stops here
    fi

    if [ $4 -ge 10 ]; then
        echo "Too many REDIS! (> 10)"
        sleep infinity # stops here
    fi

    REDIS_HOST=$3
    NUM_REDIS=$4
    if [ ${NUM_REDIS} -le 0 ]; then
        REDIS_PORT=null
        echo "NO REDIS."
    else
        case "$2" in
            1) REDIS_PORT=$(seq -s '|' 6381 $((${NUM_REDIS}+6380))) # redis-1 <-> mms-1
            ;;
            2) REDIS_PORT=$(seq -s '|' 6391 $((${NUM_REDIS}+6390))) # redis-2 <-> mms-2
            ;;
        esac
        for PORT in ${REDIS_PORT[*]}; do # push images to all redis
            python3 redis-cache-file.py image ${REDIS_HOST} ${PORT}
        done
        echo "REDIS_HOST: $REDIS_HOST"
        echo "REDIS_PORT: $REDIS_PORT"
    fi


    if [ -z "$5" ]; then
        MODEL="https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar"
    else
        case "$5" in
            1) MODEL="squeezenet"
            MODEL_PATH="https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar" ;;
            2) MODEL="resnet-152"
            MODEL_PATH="https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152.mar" ;;
            3) MODEL="inception-bn"
            MODEL_PATH="https://s3.amazonaws.com/model-server/model_archive_1.0/inception-bn.mar" ;;
        esac
    fi
    echo "MODEL: $MODEL"

    if [ -z "$6" ]; then
        BATCH_SIZE=512
    else
        BATCH_SIZE=$6
    fi
    echo "BATCH_SIZE: $BATCH_SIZE"

    mxnet-model-server --stop
    mxnet-model-server --start --models ${MODEL}=${MODEL_PATH}

    python3 aiohttp-image-redis-8079.py --redis_host $REDIS_HOST --redis_port $REDIS_PORT --model $MODEL --batch_size $BATCH_SIZE
else
    echo "mms <1,2> <redis_host> <#redis> <1,2,3> <batch_size>"
    sleep infinity
fi
