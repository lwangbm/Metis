import sys
import os
import argparse
sys.path.append("/Users/ourokutaira/Desktop/Metis")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import time
import os
from cluster_env import LraClusterEnv   # 81
from PolicyGradientHighlevel import PolicyGradient
from simulator.simulator import Simulator


"""
'--batch_choice': 0, 1, 2, ``` 30
'--batch_set_size': 200, 300, 400
python3 VanillaRL81nodes.py --batch_set_size 200 --batch_choice 0
"""

hyper_parameter = {
        'batch_set_choice': -1,
        'batch_C_numbers': -1
}
params = {
        'batch_size': 10,
        'epochs': 800000,
        'path': "VanillaRL_81_" + str(hyper_parameter['batch_C_numbers']),
        'recover': False,
        'number of containers': -1,
        'learning rate': 0.001,
        'nodes per group': 81,
        'number of nodes in the cluster': 81,  # 81
        'replay size': 10,
        'container_limitation per node': 8   # 81
    }


def choose_action_external_knowledge(observation_new_list, app_index):
    # external: new action-chosen method with external knowledge

    observation_list = observation_new_list[0, 0:-7].reshape(params['nodes per group'], 7) #[0:27]

    #: first, we select the nodes with min No. of containers
    index_min_total = np.where(observation_list.sum(1) == observation_list.sum(1).min())
    observation_list_second = observation_list[index_min_total]

    #: second, we select the nodes with min No. of app_index
    index_min_appindex = np.where(observation_list_second[:,int(app_index)] == observation_list_second[:, int(app_index)].min())
    # node = index_min_total[0][index_min_appindex[0][0]]
    node = index_min_total[0][np.random.choice(index_min_appindex[0])]

    return node, []


def handle_constraint(observation, NUM_NODES):
    """
    for all these 3 nodes, we kick out those who violate the hard constraint,
    in order to achieve this, we replace these 'bad' nodes with 'good' ones, and maintain their mapping relations,
    this is because we wanna keep the No. of node-candidates as 3, so we can remain other parts of code unchanged

    :return:
    :observation: new observation
    :mapping_index: mapping relations (good_node's is itself, bad_node -> good_node)
    """
    observation_original = observation.copy()
    mapping_index = []
    # TODO: we could add more constraints here
    list_check = observation[:, :].sum(1) > params['container_limitation per node'] - 1   # >81

    if sum(list_check) == NUM_NODES:
        return [],[]

    good_index = np.where(list_check == False)[0]
    length = len(good_index)
    index_replace = 0
    for node in range(NUM_NODES):
        if list_check[node]:  # bad node
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
    env = LraClusterEnv(num_nodes=NUM_NODES)
    batch_size = params['batch_size']
    ckpt_path = "./checkpoint/" + params['path'] + "/model.ckpt"
    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    nodes_per_group = int(params['nodes per group'])
    training_times_per_episode = 1  # TODO: if layers changes, training_times_per_episode should be modified
    sim = Simulator()

    """
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * env.NUM_APPS + env.NUM_APPS)  #: 3*9+1 = 28
    RL_1 = PolicyGradient(n_actions=n_actions, n_features=n_features, learning_rate=params['learning rate'], suffix='1')

    """
    Training
    """
    start_time = time.time()
    highest_value_time = 0.0
    print("start time!")
    global_start_time = start_time
    observation_episode_1, action_episode_1, reward_episode_1 = [], [], []
    observation_optimal_1, action_optimal_1, reward_optimal_1 = [], [], []
    highest_tput = 0.1
    epoch_i = 0
    optimal_range = 1.02
    allocation_optimal = []

    def store_episode_1(observations, actions):
        observation_episode_1.append(observations)
        action_episode_1.append(actions)
    source_batch, index_data, embedding = batch_data()  # index_data = [0,1,2,0,1,2]

    while epoch_i < params['epochs']:

        observation = env.reset().copy()  # (9,9)

        """
        Episode
        """
        for inter_episode_index in range(NUM_CONTAINERS):
            observation, mapping_index = handle_constraint(observation, NUM_NODES)
            assert len(mapping_index) > 0

            """
            first layer
            """
            observation_first_layer = np.empty([0, env.NUM_APPS], int)
            number_of_first_layer_nodes = int(NUM_NODES / nodes_per_group)  # 9
            for i in range(nodes_per_group):
                observation_new = np.sum(observation[i * number_of_first_layer_nodes:(i + 1) * number_of_first_layer_nodes], 0).reshape(1, -1)
                observation_first_layer = np.append(observation_first_layer, observation_new, 0)
            observation_first_layer[:, index_data[inter_episode_index]] += 1
            observation_first_layer = np.append(observation_first_layer, embedding[index_data[inter_episode_index]]).reshape(1,-1)
            action_1, prob_weights = RL_1.choose_action(observation_first_layer)
            """
            final decision
            """
            final_decision = action_1 * number_of_first_layer_nodes #+ action_2 * number_of_second_layer_nodes #+ action_3 * number_of_third_layer_nodes + action_4 * number_of_fourth_layer_nodes
            appid = index_data[inter_episode_index]
            observation_ = env.step(mapping_index[final_decision], appid)

            store_episode_1(observation_first_layer, action_1)
            observation = observation_.copy()  # (9,9)

        """
        After an entire allocation, calculate total throughput, reward
        """
        tput_state = env.get_tput_total_env()
        tput = (sim.predict(tput_state.reshape(-1, env.NUM_APPS)) * tput_state).sum() / NUM_CONTAINERS
        RL_1.store_tput_per_episode(tput, epoch_i, time.time() - global_start_time)

        assert (np.sum(env.state, axis=1) <= params['container_limitation per node']).all()
        assert sum(sum(env.state)) == NUM_CONTAINERS

        reward_ratio = (tput - highest_tput)
        reward_episode_1 = [reward_ratio] * len(observation_episode_1)
        RL_1.store_training_samples_per_episode(observation_episode_1, action_episode_1, reward_episode_1)

        """
        check_tput_quality(tput)
        """
        if highest_tput < tput:
            highest_tput_original = highest_tput
            optimal_range_original = optimal_range
            highest_tput = tput
            highest_value_time = time.time() - start_time
            allocation_optimal = env.state
            observation_optimal_1, action_optimal_1, reward_optimal_1 = [], [], []
            observation_optimal_1.extend(observation_episode_1)
            action_optimal_1.extend(action_episode_1)
            reward_optimal_1.extend(reward_episode_1)
            optimal_range = min(1.02, highest_tput / (highest_tput_original / optimal_range_original))
        observation_episode_1, action_episode_1, reward_episode_1 = [], [], []

        """
        Each batch, RL.learn()
        """
        if (epoch_i % batch_size == 0) & (epoch_i > 1):

            RL_1.learn(epoch_i, 0.0, True)
        """
        checkpoint, per 1000 episodes
        """

        if (epoch_i % 500 == 0) & (epoch_i > 1):
            count_size = (len(reward_optimal_1) / (NUM_CONTAINERS * training_times_per_episode))
            print("\n epoch: %d, highest tput: %f, times: %d" % (epoch_i, highest_tput, count_size))
            print("spending time: %f"%(time.time() - global_start_time))
            RL_1.save_session(ckpt_path)
            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode), time=np.array(RL_1.time_persisit),
                     highest_value_time=highest_value_time, highest_tput=highest_tput,
                     allocation_optimal=allocation_optimal, entropy=np.array(RL_1.entropy_persist))

        epoch_i += 1


def indices_to_one_hot(data, nb_classes):  #separate: embedding
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

    # separate: embedding
    nb_classes = 7
    data = range(7)
    embedding = indices_to_one_hot(data, nb_classes)

    return rnd_array, index_data, embedding

def make_path(dirname):

    if not os.path.exists("./checkpoint/" + dirname):
        os.mkdir("./checkpoint/"+ dirname)
        print("Directory ", "./checkpoint/" + dirname, " Created ")
    else:
        print("Directory ", "./checkpoint/" + dirname, " already exists")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_set_size', type=int)
    parser.add_argument('--batch_choice', type=int)
    args = parser.parse_args()
    hyper_parameter['batch_set_choice'] = args.batch_choice
    hyper_parameter['batch_C_numbers'] = args.batch_set_size
    params['path'] = "VanillaRL_81_" + str(hyper_parameter['batch_set_choice']) + "_" + str(hyper_parameter['batch_C_numbers'])
    params['number of containers'] = hyper_parameter['batch_C_numbers']
    make_path(params['path'])

    train(params)

if __name__ == "__main__":
    main()
