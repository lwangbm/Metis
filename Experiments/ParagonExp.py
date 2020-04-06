import numpy as np
import time
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Experiments.cluster_env import LraClusterEnv
from Experiments.paragon.paragon import ParagonScheduler
import matplotlib.pyplot as plt

"""
'--batch_choice': 0, 1, 2, ``` 30
'--batch_set_size': 200, 300, 400
python3 ParagonExp.py --batch_set_size 200 --batch_choice 0 --size medium --verbose
python3 ParagonExp.py --batch_set_size 2000 --batch_choice 0 --size large
"""

hyper_parameter = {
        'batch_set_choice': -1,
        'batch_C_numbers': -1,
}

def handle_constraint(observation, NUM_NODES, container_limitation_per_node):
    """
    for all these 9 nodes, we kick out those who violate the hard constraint,
    in order to achieve this, we replace these 'bad' nodes with 'good' ones, and maintain their mapping relations,
    this is because we wanna keep the No. of nodes candidates as 9, so we can remain other parts of code unchanged

    :return:
    :observation: new observation
    :mapping_index: mapping relations (good_node's is itself, bad_node -> good_node)
    """

    observation_original = observation.copy()
    mapping_index = []

    list_check = observation[:, :].sum(1) > container_limitation_per_node   # >8

    # all the nodes have been fulfilled, will not occur actually, since we only have 45
    if sum(list_check) == NUM_NODES:
        return [],[]

    good_index = np.where(list_check == False)[0]
    length = len(good_index)
    index_replace = 0
    for node in range(NUM_NODES):
        if list_check[node]:  # bad node
            # index_this_replace = good_index[np.random.randint(length)]
            index_this_replace = good_index[index_replace % length]
            index_replace += 1
            observation[node] = observation_original[index_this_replace]
            mapping_index.append(index_this_replace)
        else:
            mapping_index.append(node)
            observation[node] = observation_original[node]

    return observation, mapping_index

def train(params):

    """
    parameters set
    """
    NUM_NODES = params['number of nodes in the cluster']
    NUM_CONTAINERS = params['number of containers']
    NCt = hyper_parameter['batch_C_numbers']
    verbose = params.get('verbose', 0)
    env = LraClusterEnv(num_nodes=NUM_NODES)
    # ckpt_path = "/Users/ourokutaira/Desktop/Scheduling_RL/lra_scheduling_simulation/Simulation_Python/results/New_simu_270/"  + params['path'] + "/model.ckpt"
    """
    Build Network
    """

    # RL = PolicyGradient(
    #     n_actions=n_actions,
    #     n_features=n_features,
    #     learning_rate=params['learning rate'])

    """
    Training
    """
    # RL.restore_session(ckpt_path)
    # source_batch, index_data = batch_data()  # index_data = [0,1,2,0,1,2]

    def indices_to_one_hot(data, nb_classes):  # separate: embedding
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def batch_data():

        npzfile = np.load("./data/batch_set_" + str(hyper_parameter['batch_C_numbers']) + '.npz')
        batch_set = npzfile['batch_set']
        rnd_array = batch_set[hyper_parameter['batch_set_choice'], :]
        index_data = []

        for i in range(7):
            index_data.extend([i] * rnd_array[i])

        print(hyper_parameter['batch_C_numbers'])
        print(hyper_parameter['batch_set_choice'])
        print(rnd_array)
        nb_classes = 7
        data = range(7)
        embedding = indices_to_one_hot(data, nb_classes)

        return rnd_array, index_data, embedding

    """
    Episode
    """

    # verbose = 1
    paragon_verbose=0
    output_alloc_array_flag=0
    output_alloc_array=[]
    output_perf_list=[]
    sensitivity = 3
    nct_out_array = []
    perf_list = []
    start_time = time.time()

    source_batch, index_data, embedding = batch_data()
    paragon = ParagonScheduler(num_nodes=NUM_NODES, rnd_array=source_batch, sim='interference_applist.csv', sensitivity=sensitivity, verbose=paragon_verbose, node_capacity=params['container_limitation per node'])

    observation = env.reset().copy()  # (9,9)
    for inter_episode_index in range(NUM_CONTAINERS):
        observation_clean = observation.copy()
        observation[:, index_data[inter_episode_index]] += 1
        # testbed: mapping bad_node -> good_node
        observation, mapping_index = handle_constraint(observation.copy(), NUM_NODES, params['container_limitation per node'])
        appid = index_data[inter_episode_index]
        chosen_node = paragon.choose_action(app_id=appid, state=observation_clean)  # observation_clean
        observation_ = env.step(mapping_index[chosen_node], appid)

        observation = observation_.copy()  # (9,9)
    """
    After an entire allocation, calculate total throughput, reward
    """

    tput = env.get_all_node_throughput()

    if verbose:
        print("Total throughput is :", tput)
        print("aver throughput is :", tput / NCt)
        print("\nFinal state is :")
        tput_breakdown = np.empty((0, env.NUM_APPS), int)

        for node_i in range(NUM_NODES):
            tput_node, tput_breakdown_single = (env.get_throughput_single_node(node_i))
            tput_breakdown = np.append(tput_breakdown, tput_breakdown_single, axis=0)
            print(env.state[node_i], "throughput: %.5f" % tput_node)
        print("\nPer-container Throughput BreakDown for each node: ")
        np.set_printoptions(precision=2)
        print(repr(tput_breakdown))

        assert ((np.sum(env.state, axis=1) <= params['container_limitation per node']).all())
        print(source_batch)

    if output_alloc_array_flag:
        nct_out_array.append(env.state)
    perf_list.append(tput / NCt)
    scheduling_time = float(time.time() - start_time)

    print("\nAverage throughput: %.4f" % np.array(perf_list).mean())
    print("Scheduling latency: %.4f sec" % (scheduling_time))
    if output_alloc_array_flag:
        output_alloc_array.append(np.array(nct_out_array))
        output_perf_list.append(np.array(perf_list) * NCt)
    
    if output_alloc_array_flag:
        np.savez('output_alloc_array_NCt{}.npz'.format(NCt), np.array(output_alloc_array), np.array(output_perf_list))

# import numpy as np
# npzfile40, npzfile60, npzfile80 = np.load('output_alloc_array_NCt40.npz'), np.load('output_alloc_array_NCt60.npz'), np.load('output_alloc_array_NCt80.npz')
# output_array = [npzfile40['arr_0'][0], npzfile60['arr_0'][0], npzfile80['arr_0'][0]]
# np.savez('paragon-300sla-1.npz', np.array(output_array))


# def batch_data():
#
#     rnd_array = np.array([int(params['number of containers'] / params['number of nodes in the cluster'])]*9)
#     index_data = []
#
#     for i in range(params['number of nodes in the cluster']):
#         index_data.extend([i] * rnd_array[i])
#
#     return rnd_array, index_data


def plot_meta(name,label_name):

    np_path = "./checkpoint/" + name + "/optimal_file_name.npz"
    npzfile = np.load(np_path)
    tput = npzfile['tputs']
    epoch = npzfile['candidate']
    window_size = 100
    tput_smooth = np.convolve(tput, np.ones(window_size, dtype=int), 'valid')
    epoch_smooth = np.convolve(epoch, np.ones(window_size, dtype=int), 'valid')
    plt.plot(1.0 * epoch_smooth / window_size, 1.0 * tput_smooth / window_size, '.', label=label_name)


def draw_graph_single(params):

    plot_meta(params["path"], "RL")

    plt.legend(loc = 4)
    plt.xlabel("episode")
    plt.ylabel("throughput")
    plt.title("54 containers ->9 nodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_set_size', type=int, default=200)
    parser.add_argument('--batch_choice', type=int, default=0)
    parser.add_argument('--size', type=str, default='medium')
    parser.add_argument('-v', '--verbose', action='store_const', const=True)
    args = parser.parse_args()
    hyper_parameter['batch_set_choice'] = args.batch_choice
    hyper_parameter['batch_C_numbers'] = args.batch_set_size

    params={}
    if 'large' in args.size:
        params['number of nodes in the cluster'] = 729 # large
        params['container_limitation per node'] = 20  # large
    else: # i.e., medium
        params['number of nodes in the cluster'] = 81 # medium
        params['container_limitation per node'] = 8  # medium
    params['verbose'] = args.verbose
    params['number of containers'] = hyper_parameter['batch_C_numbers']
    train(params)


if __name__ == "__main__":
    main()
