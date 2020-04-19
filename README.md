Metis
====================
Metis: Learning to Schedule Long-Running Applications in Shared Container Clusters with at Scale


This repository contains the tensorflow implementation for reinforcement learning based Long Running Application scheduling with Hierarchical Reinforcement Learning (HRL). 



## Dependencies
1. Python 3.5 or above
2. Tensorflow 1.12.0
3. scipy
4. numpy
5. pandas
6. matplotlib
7. sklearn
8. Matlab 2019b or above

## Installation
### Install Requirements
```
git clone https://github.com/Metis-RL-based-container-sche/Metis.git
```
### Clone from Github source:
```
pip install -r requirement.txt
```

## Content

Our project includes three parts: 
1. [Cluster](https://github.com/Metis-RL-based-container-sche/Metis/tree/master/Cluster): Implementation of our seven real-world LRAs that exhibit inter-container interferences.
2. [Experiments](https://github.com/Metis-RL-based-container-sche/Metis/tree/master/Experiments): Metis scheduling workflow based on our real-world LRA setting.

# Experiment workflow

## Real-World LRA cluster setup and profiler establishment

1. Launch a cluster of tens of nodes on [Amazon EC2](https://aws.amazon.com/ec2) consisting of at least 1 manager node, several worker nodes, and some client nodes. Record the public DNS address or ip address of each node and make sure the manager and client nodes are accessible through the SSH key `~/.ssh/id_rsa`.
    ```bash
    export MANAGER=ec2-xxx-xxx-xxx-100.us-west-2.compute.amazonaws.com
    export WORKER1=ec2-xxx-xxx-xxx-101.us-west-2.compute.amazonaws.com
    # ......
    export CLIENT1=ec2-xxx-xxx-xxx-201.us-west-2.compute.amazonaws.com
    # ......

    $ ssh ubuntu@$MANAGER
    (manager)$ git clone https://github.com/Metis-RL-based-container-sche/Metis.git
    ```

2. If Docker has not been installed, install Docker on each manager, worker, and client node:

    ```bash
    (manager)$ sudo Cluster/scripts/install_docker.sh
    (worker1)$ sudo Cluster/scripts/install_docker.sh
    # ......
    ```

3. Coordinating manager and worker nodes through Docker Swarm to form a Swarm cluster.

    ```bash
    $ ssh $MANAGER docker swarm init

    # Output
    To add a worker to this swarm, run the following command:

        docker swarm join --token SWMTKN-1-1bu27zw7lzh6pnu1l0981bu1m6nqq2pcgk25kovuh565319cah-8480smxu3kp7cj5nfkck2itax 192.168.99.100:2377
    ```

    Then execute the aforementioned command on each worker. 

    ```bash
    $ ssh $WORKER1 docker swarm join --token SWMTKN-1-1bu27zw7lzh6pnu1l0981bu1m6nqq2pcgk25kovuh565319cah-8480smxu3kp7cj5nfkck2itax 192.168.99.100:2377

    # Output
    This node joined a swarm as a worker.
    ```

4. Build docker images of different workloads on the manager node.

    ```bash
    (manager)$ cd Cluster/workloads
    (manager)$ ./build-all.sh
    ```

5. Launch certain workloads on certain worker node from the manager node.

    ```bash
    (manager)$ cd Cluster/scripts
    # Launching 1 container for each workload on WORKER1:
    # '0' indicates idle; '1' ~ '7' indicate different workloads.
    (manager)$ ./service-launching.sh $WORKER1 0 1 2 3 4 5 6 7
    # e.g., launch 3 workload-1, 2 workload-2, and 1 workload-3 container:
    # ./service-launching.sh $WORKER2 0 0 1 1 1 2 2 3
    ```

6. Sending requests from client node to the worker node

    ```bash
    $ ssh ubuntu@$CLIENT1
    (client1)$ git clone https://github.com/Metis-RL-based-container-sche/Metis.git
    (client1)$ cd Cluster/scripts
    # export WORKER1=ec2-xxx-xxx-xxx-101.us-west-2.compute.amazonaws.com
    # Send single request for testing
    (client1)$ curl $WORKER1:8081
    # Or pressure the application with locust or other tools, e.g.,
    (client1)$ cd Cluster/scripts
    (client1)$ python3 parallel_locust.py $WORKER1
    # Default log path lies in Cluster/scripts/log
    ```

7. (Optional) Collecting performance benchmark datasets through automatically deploying, profiling, terminating, and re-deploying.

    ```bash
    # Here the "0-1-2-3-4-5-6-7" or "0-0-1-1-1-2-2-3" means combination
    # of different workloads (as described step 5).
    (manager)$ ./profiling-go.sh $WORKER1 "0-1-2-3-4-5-6-7 0-0-1-1-1-2-2-3 0-0-0-0-0-0-0-7 0-0-0-0-1-1-1-7 2-4-4-6-6-6-6-7"
    ```

8. Terminate the swarm cluster on the manager node.

    ```bash
    (manager)$ docker swarm leave --force
    ```

## Metis: RL model training for the real-world LRA cluster

1. First check the data collected by [Real-World LRA cluster](https://github.com/Metis-RL-based-container-sche/Metis/tree/master/testbed) 
is stored in the folder:

    ```
    $ cd Experiments/
    $ ls ./simulator/datasets/
    ```
    ```
    ***_sample_collected.npz 
    ```
   
2. Train sub-schedulers in a 27-node sub-cluster: 
    ```
    $ cd Experiments/
    $ ./shell/TrainSubScheduler.sh
    ```
    Output: the well-trained sub-scheduler models, as well as corresponding log files will be store in the folder:
    ```
    $ ls ./checkpoint/
    ```
    ```
    subScheduler_*/  

    ```


3. High-level model training based on previously well-trained sub-schedulers.

    (0) Check the sub-scheduler models are stored in the folder:
    ```
    $ cd Experiments/
    $ ls ./checkpoint/
    ```
    ```
    subScheduler_*/   
    ```
    Check the container batches data is stored in the folder or create your own batches:
    ```
    $ ls ./data
    ```
    ```
    batch_set_200.npz batch_set_300.npz batch_set_400.npz
    batch_set_1000.npz batch_set_2000.npz batch_set_3000.npz
    ```
    
    (1) High-level training in a medium-sized cluster of 81 nodes:
    ```
    $ ./shell/RunHighLevelTrainingMedium.sh 200
    $ ./shell/RunHighLevelTrainingMedium.sh 300
    $ ./shell/RunHighLevelTrainingMedium.sh 400
    ```
    Output: the training log files including the RPS, placement matrix, training time duration .etc will be store in the folder:
    ```
    $ ls ./checkpoint/
    ```
    ```
    81nodes_*_*/
    ```
    
    (2) High-level training in a large cluster of 729 nodes:
    ```
    $ ./shell/RunHighLevelTrainingLarge.sh 1000
    $ ./shell/RunHighLevelTrainingLarge.sh 2000
    $ ./shell/RunHighLevelTrainingLarge.sh 3000
    ```
    Output: the training log files including the RPS, placement matrix, training time duration .etc will be store in the folder:
    ```
    $ ls ./checkpoint/
    ```
     ```
    729nodes_*_*/
     ```

## Baseline Method: Vanilla RL

Vanilla RL is built directly upon Policy Gradient without our Hierarchical designs. 

1. High-level training in a medium-sized cluster of 81 nodes:

    ```
    $ ./shell/RunVanillaRLMedium.sh 200
    ```
    Output: the training log files including the RPS, placement matrix, training time duration, etc will be store in the folder:
    ```
    $ ls ./checkpoint/
    ```
    ```
    Vanilla_81_*_*/
    ```


## Baseline Method: Divide-Conquer (DC)

DC Method does not use sub-schedulers. Our code below shows its behaviors in a medium-sized cluster of 81 nodes. Each cluster is hierarchically divided into three subsets.

1. High-level training in a medium-sized cluster of 81 nodes:

    ```
    $ ./shell/RunDCMedium.sh 200
    ```
    Output: the training log files including the RPS, placement matrix, training time duration, etc will be store in the folder:
    ```
    $ ls ./checkpoint/
    ```
    ```
    DC_81_*_*/
    ```

    
## Baseline Method: Medea

Medea is implemented using Matlab, due to its outstanding performance in solving the Integer Linear Programming (ILP) Problem.

1. Generate the performance-constraints used in Medea:

    ```
    $ cd Experiments
    $ ./shell/GenerateInterference.sh
    $ ls
    ```
    
    ```
    interference_applist.csv interference_rpslist.csv
    ```
2. Run Medea in the folder:
    ```
    $ cd testbed/Medea
    $ Matlab Medea.m
    ```
    Output: the scheduling decision log files including the allocation matrix, constraint violations, time duration .etc will be store in the folder.



## Baseline Method: Paragon

Paragon is re-implemented in Python. For the sake of fair comparison, we feed it with the full interference matrix information as Medea.

1. Make sure `interference_applist.csv` has been generated in former Medea setup:

    ```
    $ ls interference_applist.csv
    ```

    Otherwise, generate the performance-constraints used in Medea:

    ```
    $ cd Experiments
    $ ./shell/GenerateInterference.sh
    $ ls interference_applist.csv
    ```

2. Run Paragon of Medium size or Large size:
    ```
    $ cd Experiments/shell
    $ # Medium size
    $ ./RunParagonMedium.sh 200
    $
    $ # Large size
    $ ./RunParagonLarge.sh 2000
    ```

    Output: the default output shows the average throughput for each testing group as well as the scheduling latency.

    For detailed output including container placement and per-container throughput breakdown for each node, please add `-v` after each python script:
    
    ```
    $ cd Experiments
    $ python3 ParagonExp.py --batch_set_size 200 --batch_choice 0 --size medium --verbose
    ```

# References
