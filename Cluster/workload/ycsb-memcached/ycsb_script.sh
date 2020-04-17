#!/bin/bash
if [ -z "$1" ]; then
    WORKLOAD="workloada"
else
    WORKLOAD=$1
fi

if [ -z "$2" ]; then
    OUTPUT_FILE="/output.txt"
else
    OUTPUT_FILE=$2
fi

ycsb_load_workload () {
    ./bin/ycsb load memcached -s -p "memcached.hosts=${SELF_HOST}" -P workloads/${WORKLOAD} -threads 2
}

ycsb_run_workload () {
    ./bin/ycsb run memcached -s -p "memcached.hosts=${SELF_HOST}" -P workloads/${WORKLOAD} -threads 2
}

ycsb_infinite_loop () {    
    for i in {1..30}; do 
        ycsb_run_workload | grep Throughput >> ${OUTPUT_FILE}
        sleep 0.5
    done
}

echo "Hi, Memcached at ${SELF_HOST}"
cd /YCSB/
ycsb_load_workload
ycsb_infinite_loop
echo "Ends"