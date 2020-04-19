#!/usr/bin/env python3
# __file__: /home/ubuntu/parallel_locust.py

# profiling-go.sh calling:
# python3 parallel_locust.py $PROFILEE $LOCUST_CLIENT=4 $LOCUST_RUMTIME=300 $LOGFOLDER $STREAM_INDEX='='

import sys
import locust.main as locust_main
from multiprocessing import Process, Pool, TimeoutError
import random
import time
import requests
import numpy as np
import os
import json

def save_latency_to_logfile_csv(latency_value, logfile):
    # take (1 / average-latency-of-99%-requests) as rps
    latency_array = np.array(latency_value)
    latency_array.sort()
    num_requests = len(latency_value)
    l0 = latency_array[0]
    l50 = latency_array[int(0.5 * num_requests)]
    l66 = latency_array[int(0.66 * num_requests)]
    l75 = latency_array[int(0.75 * num_requests)]
    l80 = latency_array[int(0.80 * num_requests)]
    l90 = latency_array[int(0.90 * num_requests)]
    l95 = latency_array[int(0.95 * num_requests)]
    l98 = latency_array[int(0.98 * num_requests)]
    l99 = latency_array[int(0.99 * num_requests)]
    l100 = latency_array[-1]
    avg = latency_array.mean()
    # avg99 = latency_array[:int(0.99 * num_requests)].mean()
    sla_latency = 750 # used to be 1000 or 500
    rps = np.searchsorted(latency_array, sla_latency) / 240 # return the index of where 1000 should be, even if 1000 does not exists, 240: running time

    title=['"Name"','"# requests"','"50%"','"66%"','"75%"',
          '"80%"','"90%"','"95%"','"98%"','"99%"',
          '"100%"','"Method"','"Name"','"# requests"','"# failures"',
          '"Median response time"','"Average response time"','"Min response time"','"Max response time"','"Average Content Size"',
          '"Requests/s"'] # 21

    content = ['"Stream"',str(num_requests),'%.1f' % l50,'%.1f' % l66,'%.1f' % l75,
            '%.1f' % l80,'%.1f' % l90,'%.1f' % l95,'%.1f' % l98,'%.1f' % l99,
            '%.1f' % l100,'"None"','"Total"',str(num_requests),str(0),
            '%.1f' % l50,'%.1f' % avg,'%.1f' % l0,'%.1f' % l100,str(0),'%.3f' % rps]

    np.savez(logfile+'.npz', latency=latency_array)

    with open(logfile, 'w') as f:
        f.write(','.join(title)+'\n')
        f.write(','.join(content))

def save_throughput_to_logfile_csv(rps_value, logfile):
    # save YCSB's rps_value into rps
    rps_array = np.array(rps_value)
    rps_value.sort()
    num_requests = len(rps_array)
    rps = rps_array.mean() / 1000  # changed into Kop/s
    
    title=['"Name"','"# requests"','"50%"','"66%"','"75%"',
    '"80%"','"90%"','"95%"','"98%"','"99%"',
    '"100%"','"Method"','"Name"','"# requests"','"# failures"',
    '"Median response time"','"Average response time"','"Min response time"','"Max response time"','"Average Content Size"',
    '"Requests/s"'] # 21
    
    content = ['"YCSB"',str(num_requests),'0' ,'0' ,'0',
            '0','0','0','0','0',
            '0','"None"','"Total"',str(num_requests),str(0),
            '0','0','0','0',str(0),'%.3f' % rps]
    
    np.savez(logfile+'.npz', rps=rps_array)

    with open(logfile, 'w') as f:
        f.write(','.join(title)+'\n')
        f.write(','.join(content))
        

def combine_dist_request_csv(logfile):
    dis_logfile = logfile+'_distribution.csv'
    req_logfile = logfile+'_requests.csv'
    out_logfile = logfile+'.csv'
    with open(dis_logfile, 'r') as dis_f:
        dl1=dis_f.readline()[:-1]
        dl2= dis_f.readline()[:-1]
    with open(req_logfile, 'r') as req_f:
        rl1=req_f.readline()
        rl2=req_f.readline()[:-1]
    ol1=dl1+','+rl1
    ol2=dl2+','+rl2
    with open(out_logfile, 'w') as out_f:
        out_f.write(ol1)
        out_f.write(ol2)
    if os.path.exists(out_logfile):
        os.remove(dis_logfile)
        os.remove(req_logfile)

def send_reqs(index, host, logfolder, is_isr, is_ycsb, is_scnt, is_stream, locust_client=4, locust_runtime=300, locust_hatchrate=50, no_web=True, only_summary=True):
    #: script-profiling.sh calling:
    #: python3 -m locust.main --host=http://$HOST:$PORT --clients=$LOCUST_CLIENT --hatch-rate=$LOCUST_HATCHRATE --no-web --run-time=$LOCUST_RUNTIME --only-summary --csv=$LOGFILE

    port = index+8081  # 8081, 8082, ..., 8089
    logfile = "{}/log-{}".format(logfolder, port)
    if is_stream or is_ycsb:
        # busy querying port:8080 port to check whether the service is online
        # once get file transferred, exit
        r = None
        while not r:
            try:
                r = requests.get('http://{}:{}/latency'.format(host, port),  timeout=10)
            except:
                pass
            time.sleep(10)
        if is_stream:
            results_json = json.loads(r.text)
            latency_value = results_json['latency_value']
            save_latency_to_logfile_csv(latency_value, logfile+'.csv')
        elif is_ycsb:
            results_json = json.loads(r.text)
            rps_value = results_json['rps_value']
            save_throughput_to_logfile_csv(rps_value, logfile+'.csv')

    else:
        sys.argv = ['locust'] # remove all the existing arguments
        sys.argv.append('--host=http://{}:{}'.format(host, port))
        sys.argv.append('--clients={}'.format(locust_client))
        sys.argv.append('--hatch-rate={}'.format(locust_hatchrate))
        sys.argv.append('--run-time={}'.format(locust_runtime))
        if no_web: sys.argv.append('--no-web')
        if only_summary: sys.argv.append('--only-summary')
        sys.argv.append('--csv={}'.format(logfile))
        #: All Processes have its own copy of the same original sys.argv
        # print(sys.argv)

        locust_main.main()
        time.sleep(5) # wait for scripts to be written into .csv
        combine_dist_request_csv(logfile)

    return port

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 parallel_locust.py $PROFILEE $LOCUST_CLIENT=4 $LOCUST_RUMTIME=300 $LOGFOLDER $ISR_INDEX='=' $YCSB_INDEX='=' $SCDT_INDEX='=' $STREAM_INDEX='='")
        exit()
    elif len(sys.argv) < 6:
        sys.argv = [sys.argv[0], sys.argv[1], 4, 300, 'log', '=', '=', '=', '=']
    
    profilee=sys.argv[1]
    locust_client=sys.argv[2]
    locust_runtime=sys.argv[3]
    logfolder=sys.argv[4]
    
    isr_index=[int(x) for x in sys.argv[5].split('=') if (len(x) > 0) ] # 5. image super resolution
    ycsb_index=[int(x) for x in sys.argv[6].split('=') if (len(x) > 0) ] # 6. ycsb clients
    scdt_index=[int(x) for x in sys.argv[7].split('=') if (len(x) > 0) ] # 7. scene detection
    stream_index=[int(x) for x in sys.argv[8].split('=') if (len(x) > 0) ] # 8. or 9., streaming "==1=2=3" or "==1=3" or "=" or ...
    
    print("ISR, YCSB, SCENE_DETECT, STREAM", isr_index, ycsb_index, scdt_index, stream_index)
    if not os.path.exists(logfolder):
        os.mkdir(logfolder)
    print("[INFO] Save into logfolder {}".format(logfolder))
    
    num_port=8
    # process implementation
    processes = []
    for x in range(num_port):
        is_isr=True if x in isr_index else False
        is_ycsb=True if x in ycsb_index else False
        is_scnt=True if x in scdt_index else False
        is_stream=True if x in stream_index else False
        processes.append(Process(target=send_reqs, args=(x, profilee, logfolder, is_isr, is_ycsb, is_scnt, is_stream, locust_client, locust_runtime)))
    print("[INFO] Start sending requests ...")
    for p in processes:
        p.start()
    #: working, waiting until all ends
    for p in processes:
        p.join(480) # timeout: 8 minutes
    print("[INFO] Profiling ends.")
