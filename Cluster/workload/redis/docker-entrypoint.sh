#!/bin/bash
set -e
if [ "$1" = 'redis' ]; then
    redis-server --daemonize yes --protected-mode no
    if [ -z "$2" ]; then
        OPTIONS=--requests 2000 --clients 1 -t get,set,incr
    else
        case "$2" in
            # 2vCPU, 8GB: around 2000ms latency
            1) OPTIONS="--requests 20000 --clients 1 -t get,set,incr" ;;
            2) OPTIONS="--requests 20000 --clients 1 -t lpush,lpop,incr" ;;
            3) OPTIONS="--requests 20000 --clients 10 -t get,set,incr,lpush,lpop,incr"
        esac
    fi
    echo "OPTIONS: $OPTIONS"
    python3 aiohttp-redis-benchmark.py $OPTIONS
elif [ "$1" = 'redis-alone' ]; then
    redis-server --protected-mode no
else
    echo "redis <1,2> OR redis-alone"
    /bin/bash
fi
