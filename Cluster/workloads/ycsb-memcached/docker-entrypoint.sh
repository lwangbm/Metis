#!/bin/bash
# set -e # should not be set if using timeout kill

# available: workloada, workloadb, ..., workloadf
if [[ -z $1 ]]; then
    WORKLOAD='workloada'
else
    WORKLOAD=$1
fi

# wait for other containers to warm-up
if [[ -z $2 ]]; then
    sleep 45
else
    sleep $1
fi

# specific log file
OUTPUT_FILE='/output.txt'

# run
timeout 280 /YCSB/ycsb_script.sh ${WORKLOAD} ${OUTPUT_FILE}

# finish, upload logs to aiohttp
sleep 2
if [[ -n ${OUTPUT_FILE} ]]; then
    python3 /YCSB/aiohttp-ycsb.py ${WORKLOAD} ${OUTPUT_FILE}
else
    echo "[WRONG] File ${OUTPUT_FILE} not found!"
fi
