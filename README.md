Metis: Learning to Schedule Long-Running Applications in Shared Container Clusters with at Scale
====================


This repository contains the tensorflow implementation for reinforcement learning based Long Running Application scheduling with Hierarchical Reinforcement Learning (HRL). 



## Dependencies
1. Python 3.5 or above
2. Tensorflow 1.12.0
3. scipy v1.1.0
4. numpy v1.17.1
5. pandas v0.20.3
6. sklearn
7. Matlab 2019b or above

## Installation
### Clone from Github source:
```
git clone https://github.com/Metis-RL-based-container-sche/Metis.git
```
### Install Requirements
```
pip install -r requirement.txt
```

## Content

Our project includes three parts: 
1. [Cluster](https://github.com/Metis-RL-based-container-sche/Metis/tree/master/Cluster): Implementation of our seven real-world LRAs that exhibit inter-container interferences.
2. [Experiments](https://github.com/Metis-RL-based-container-sche/Metis/tree/master/Experiments): Metis scheduling workflow based on our real-world LRA setting.

# Experiment workflow

## Real-World LRA cluster setup and profiler establishment


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
