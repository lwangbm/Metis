# Cluster

- [Cluster](#cluster)
    - [Scripts](#scripts)
    - [Real-world Applications](#real-world-applications)
        - [1. Redis](#1-redis)
        - [2. MXNet Model Serving](#2-mxnet-model-serving)
        - [3. Model Checksum](#3-model-checksum)
        - [4. Image Super Resolution (ISR)](#4-image-super-resolution-isr)
        - [5 & 6. YCSB](#5--6-ycsb)
        - [7. Video Scene Detection](#7-video-scene-detection)


The scripts locate in folder `scripts` while the workloads (i.e., Real-world applications) locate in folder `workloads`.


## Scripts

- `install_docker.sh`: install docker on each node (Ubuntu).

- `parallel_locust.py`: generate requests for load testing towards applications, using [Python Locust](https://locust.io). Requires `locustfile.py` in the same folder.

   ```bash
   python3 parallel_locust.py $WORKER1
   ```

- `profiling-go.sh`: collect performance benchmark datasets through automatically deploying, profiling, terminating, and re-deploying. Requires `script-launching-8-slot.sh` in the same folder.

   ```bash
   ./profiling-go.sh $WORKER1 '6-6-6-6-7-7-7-7 0-0-0-1-6-6-6-8' $WORKER1_INSTANCE_ID
   ```

- `service-launching.sh`: (on Docker Swarm manager) launch certain workloads on certain worker node from the manager node.

    ```bash
    ./service-launching.sh $WORKER1 0 1 2 3 4 5 6 7
    ```


## Real-world Applications

All workloads are encapsulated into docker images, exposing `8080` port for http requests. Please refer to README.md in each subfolder for detailed testing commands and outputs.

1. Redis
2. MXNet Model Server
3. Model Checksum
4. Image Super Resolution
5. & 6. Yahoo! Cloud Streaming Benchmark A & B
7. Video Scene Detection


### 1. Redis
Having multiple functions:
1. On its own profiling, running command line workload: `redis-benchmark --requests 20000 --clients 1 -t lpush,lpop,incr`
2. Auxiliary of others: "MXNet Model Serving", "Flink", "Storm", etc.

References: https://redis.io/topics/benchmarks


### 2. MXNet Model Serving
Having two working mode:
1. Standing alone: once receiving a request, it pulls a jpg image from [Amazon S3](https://s3.amazonaws.com/model-server/inputs/kitten.jpg), sends it to "MXNet Model Server" running in the background, and reply results to the client.
2. Co-located with "Redis":

References: https://github.com/awslabs/mxnet-model-server


### 3. Model Checksum

Description: Checking the integrity of large model files (>100MB, i.e., resnet50 model) stored on a remote server with SHA256 hash function.

Workflow: once receiving a request, it will download a file, cache in memory, calculate its checksum.

Performance: 1.42 requests/sec, median response time 2.3 sec.

References: https://help.ubuntu.com/community/HowToSHA256SUM


### 4. Image Super Resolution (ISR)

Description: As an online service, ISR receives users’ uploaded images, uses deep neural networks to upscale and improve the quality of these low-resolution images, and returns the images with higher resolution (4x larger in size).

Workflow: receiving users’ image input (34KB .png file), ISR deploys Residual Dense Network with TensorFlow as backend to processes the images in an online manner and returns a .png file of 136KB.

Performance: 0.365 requests/sec, median response time 10 sec.

References: https://github.com/idealo/image-super-resolution


### 5 & 6. YCSB
Executes a bunch of standard query workloads (1600+ requests/sec for <2vCPU, 8GB MEM>) for certain period of time, pressuring Memcached deployed on each machine as infrastructure.

References: https://github.com/brianfrankcooper/YCSB


### 7. Video Scene Detection

Description: it detects scene changes in videos (e.g., movies) and splits the video into temporal segments. A good detector not only cuts the files by comparing the intensity/brightness between each frame but also analyzes the contents.

Workflow: it gets a 2min video (~45MB) from a remote server (the same one as 4. Model Checksum), down-scales it (for processing efficiency), process with content-aware detection algorithm, and returns a list of timepoints where the video could be separated.

Performance: 2.31 requests/sec, median response time 1.2 sec.

References: http://py.scenedetect.com/
