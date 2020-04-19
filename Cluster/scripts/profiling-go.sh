#!/bin/bash
set -e
# workload order:
# 1. redis for MMS
# 2. MMS
# 3. checksum
# 4. YCSB-A
# 5. YCSB-B
# 6. video scene detect
# 7. image super resolution

if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 PROFILEE_IP '6-6-6-6-7-7-7-7 0-0-0-1-6-6-6-8 ......' (PROFILEE_INSTANCE_ID)"
    exit
else
    PROFILEE=$1
    read -a PROFILE_SERIES <<< $2  # "6-6-6-6-7-7-7-7 0-0-0-1-6-6-6-8 ......"
fi

# e.g.,
# PROFILEE=ec2-34-214-19-102.us-west-2.compute.amazonaws.com
# PROFILEE_INSTANCE_ID=i-006cc1c18d6289d45
# PROFILE_SERIES=(6-6-6-6-7-7-7-7 0-0-0-1-6-6-6-8)

echo "[INFO] PROFILEE: $PROFILEE"
ssh -i ~/.ssh/id_rsa ubuntu@${PROFILEE} "sudo /etc/init.d/redis-server stop"
ssh -i ~/.ssh/id_rsa ubuntu@${PROFILEE} "sudo wondershaper -c -a ens5; sudo wondershaper -a ens5 -u 4194304 -d 4194304"
scp -i ~/.ssh/id_rsa script-launching-8-slot.sh ubuntu@${PROFILEE}:/home/ubuntu/
echo "[INFO] Auto-profiling ${#PROFILE_SERIES[@]} combinations: ${PROFILE_SERIES[@]}"

LOGFOLDER=microbench_log/$PROFILEE
mkdir -p $LOGFOLDER
echo "[INFO] Log saved into $LOGFOLDER"
# LOCUST_DIR=/home/ubuntu/locust

# Launch NGINX server
docker run --rm -d -p 8631:80 aiohttp/nginx
NGINX_HOST="$(hostname -i):8631"

# SLEEP_TIME=$((1 + $RANDOM % 10))
# echo "Random sleep $SLEEP_TIME sec to avoid network congestion"
# sleep $SLEEP_TIME # random sleep from 1 sec to 10 sec

for i in "${PROFILE_SERIES[@]}"; do
    PROFILEE_WARMUP_TIME=30
    echo "[INFO] Profiling applications: $i"
    PROFILE_COMMAND="/home/ubuntu/script-launching-8-slot.sh ${NGINX_HOST}"
    J_INDEX=0

    ISR_INDEX='='
    YCSB_INDEX='='
    SCDT_INDEX='='
    STREAM_INDEX='='
    
    IFS='-'; read -ra WORKLOAD <<< "$i"
    for j in "${WORKLOAD[@]}"; do
        if [[ j -eq 4 ]] || [[ j -eq 5 ]]; then
            YCSB_INDEX="${YCSB_INDEX}=${J_INDEX}" # string concat
        fi
        if [[ j -eq 6 ]]; then
            SCDT_INDEX="${SCDT_INDEX}=${J_INDEX}" # string concat
        fi
        if [[ j -eq 7 ]]; then
            ISR_INDEX="${ISR_INDEX}=${J_INDEX}" # string concat
        fi
        if [[ j -eq 8 ]] || [[ j -eq 9 ]]; then  # num 8 and num 9 are streaming workloads
            STREAM_INDEX="${STREAM_INDEX}=${J_INDEX}" # string concat
        fi
        J_INDEX=$((J_INDEX + 1)) # J_INDEX++
        PROFILE_COMMAND="${PROFILE_COMMAND} ${j}"
    done
    IFS=' '

    echo "[INFO] Stopping existing dockers (if any) ..."
    ssh -i ~/.ssh/id_rsa ubuntu@${PROFILEE} 'docker stop $(docker container ls -a -q)'
    sleep 2

    echo "[INFO] Launching containers: $2 at $1"
    echo "PROFILE_COMMAND: $PROFILE_COMMAND"
    ssh -i ~/.ssh/id_rsa ubuntu@${PROFILEE} "$PROFILE_COMMAND"

    echo "[INFO] Wait $PROFILEE_WARMUP_TIME sec for Profilee to warm up ..."
    sleep $PROFILEE_WARMUP_TIME

    LOCUST_CLIENT=4
    LOCUST_RUMTIME=300 # 5 min
    SENDER_SCRIPT="parallel_locust.py"
    timeout -k 480 480 python3 $SENDER_SCRIPT $PROFILEE $LOCUST_CLIENT $LOCUST_RUMTIME $LOGFOLDER $ISR_INDEX $YCSB_INDEX $SCDT_INDEX $STREAM_INDEX # timeout: 8 min

    echo "[INFO] Profiling DONE, collecting logs ..."
    sleep 2

    mkdir -p ${LOGFOLDER}/${i} && mv ${LOGFOLDER}/*.csv ${LOGFOLDER}/${i}/
    mv ${LOGFOLDER}/*.npz ${LOGFOLDER}/${i}/

    echo "[INFO] End of the Loop."
    sleep 2
done

# Automatically terminate nodes on EC2 once finished.
if [[ -n $3 ]]; then
    PROFILEE_INSTANCE_ID=$3
    { # try: stop
        aws ec2 stop-instances --instance-ids $PROFILEE_INSTANCE_ID
    } || { # catch: terminate (only for spot instances), won't execute if "try" works
        aws ec2 terminate-instances --instance-ids $PROFILEE_INSTANCE_ID
    }
fi
