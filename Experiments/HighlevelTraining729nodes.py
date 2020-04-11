import sys
import os
import argparse
sys.path.append("/Users/ourokutaira/Desktop/Metis")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import time
import os
from highlevel_env import LraClusterEnv
from PolicyGradientHighlevel import PolicyGradient
import copy

"""
'--batch_choice': 0, 1, 2, ``` 30
'--batch_set_size': 1000, 2000, 3000
python3 HighlevelTraining729nodes.py --batch_set_size 1000 --batch_choice 0
"""

hyper_parameter = {
        'batch_set_choice': -1,
        'batch_C_numbers': -1
}
params = {
        'batch_size': 10,
        'epochs': 2000,
        'path': "729nodes_" + str(hyper_parameter['batch_set_choice']) + "_" + str(hyper_parameter['batch_C_numbers']),
        'recover': False,
        'number of containers': -1,
        'learning rate': 0.015,
        'nodes per group': 3,
        'number of nodes in the cluster': 27,  # 81
        'replay size': 10,
        'container_limitation per node': 200   # 81
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
    env = LraClusterEnv(num_nodes=NUM_NODES)
    batch_size = params['batch_size']
    ckpt_path = "./checkpoint/" + params['path'] + "/model.ckpt"
    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    nodes_per_group = int(params['nodes per group'])
    replay_size = params['replay size']
    training_times_per_episode = 1  # TODO: if layers changes, training_times_per_episode should be modified

    """
    Build Network
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * env.NUM_APPS + env.NUM_APPS)  #: 3*9+1 = 28
    RL_1 = PolicyGradient(n_actions=n_actions, n_features=n_features, learning_rate=params['learning rate'], suffix='1')
    RL_2 = PolicyGradient(n_actions=n_actions, n_features=n_features, learning_rate=params['learning rate'], suffix='2')
    RL_3 = PolicyGradient(n_actions=n_actions, n_features=n_features, learning_rate=params['learning rate'], suffix='3')

    """
    Training
    """
    start_time = time.time()
    highest_value_time = 0.0
    print("start time!")
    global_start_time = start_time
    observation_episode_1, action_episode_1, reward_episode_1 = [], [], []
    observation_optimal_1, action_optimal_1, reward_optimal_1 = [], [], []

    observation_episode_2, action_episode_2, reward_episode_2 = [], [], []
    observation_optimal_2, action_optimal_2, reward_optimal_2 = [], [], []

    observation_episode_3, action_episode_3, reward_episode_3 = [], [], []
    observation_optimal_3, action_optimal_3, reward_optimal_3 = [], [], []

    observation_demo_1, action_demo_1, reward_demo_1 = [], [], []
    observation_demo_2, action_demo_2, reward_demo_2 = [], [], []
    observation_demo_3, action_demo_3, reward_demo_3 = [], [], []

    highest_tput = 1.0
    highest_demo_tput = 1.0
    epoch_i = 0
    optimal_range = 1.02
    allocation_optimal = []
    allocation_demo = []
    # TODO: delete this range
    # external
    use_external_knowledge = True
    entropy_weight = 0.1
    # external
    external_tput_record = []

    def store_episode_1(observations, actions):
        observation_episode_1.append(observations)
        action_episode_1.append(actions)

    def store_episode_2(observations, actions):
        observation_episode_2.append(observations)
        action_episode_2.append(actions)

    def store_episode_3(observations, actions):
        observation_episode_3.append(observations)
        action_episode_3.append(actions)


    source_batch, index_data, embedding = batch_data()  # index_data = [0,1,2,0,1,2]

    while epoch_i < params['epochs']:

        observation = env.reset().copy()  # (9,9)

        """
        Episode
        """
        for inter_episode_index in range(NUM_CONTAINERS):
            app_index = index_data[inter_episode_index]

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
            if use_external_knowledge:
                action_1, prob_weights = choose_action_external_knowledge(observation_first_layer, app_index)
            else:
                action_1, prob_weights = RL_1.choose_action(observation_first_layer)

            """
            second layer
            """
            observation_copy = observation
            observation_copy = observation_copy[action_1 * number_of_first_layer_nodes: (action_1 + 1) * number_of_first_layer_nodes]
            number_of_second_layer_nodes = int(number_of_first_layer_nodes / nodes_per_group)  # 27/3 = 9
            observation_second_layer = np.empty([0, env.NUM_APPS], int)
            for i in range(nodes_per_group):
                observation_new = np.sum(observation_copy[i * number_of_second_layer_nodes:(i + 1) * number_of_second_layer_nodes], 0).reshape(1, -1)
                observation_second_layer = np.append(observation_second_layer, observation_new, 0)
            observation_second_layer[:, index_data[inter_episode_index]] += 1
            observation_second_layer = np.append(observation_second_layer, embedding[index_data[inter_episode_index]]).reshape(1,-1)
            if use_external_knowledge:
                action_2, prob_weights = choose_action_external_knowledge(observation_second_layer, app_index)
            else:
                action_2, prob_weights = RL_2.choose_action(observation_second_layer)

            """
            third layer
            """
            observation_copy = observation_copy[action_2 * number_of_second_layer_nodes: (action_2 + 1) * number_of_second_layer_nodes]
            number_of_third_layer_nodes = int(number_of_second_layer_nodes / nodes_per_group)  # 9/3 = 3
            observation_third_layer = np.empty([0, env.NUM_APPS], int)
            for i in range(nodes_per_group):
                observation_new = np.sum(observation_copy[i * number_of_third_layer_nodes:(i + 1) * number_of_third_layer_nodes], 0).reshape(1, -1)
                observation_third_layer = np.append(observation_third_layer, observation_new, 0)
            observation_third_layer[:, index_data[inter_episode_index]] += 1
            observation_third_layer = np.append(observation_third_layer, embedding[index_data[inter_episode_index]]).reshape(1,-1)

            if use_external_knowledge:
                action_3, prob_weights = choose_action_external_knowledge(observation_third_layer, app_index)
            else:
                action_3, prob_weights = RL_3.choose_action(observation_third_layer)

            """
            final decision
            """
            final_decision = action_1 * number_of_first_layer_nodes + action_2 * number_of_second_layer_nodes + action_3 * number_of_third_layer_nodes
            appid = index_data[inter_episode_index]
            observation_ = env.step(mapping_index[final_decision], appid)

            store_episode_1(observation_first_layer, action_1)
            store_episode_2(observation_second_layer, action_2)
            store_episode_3(observation_third_layer, action_3)

            observation = observation_.copy()  # (9,9)

        """
        After an entire allocation, calculate total throughput, reward
        """
        tput = env.get_tput_total_env() / NUM_CONTAINERS
        if use_external_knowledge:
            external_tput_record.append(tput)

        RL_1.store_tput_per_episode(tput, epoch_i, time.time() - global_start_time)

        assert (np.sum(env.state, axis=1) <= params['container_limitation per node']).all()
        assert sum(sum(env.state)) == NUM_CONTAINERS

        reward_ratio = (tput - highest_tput)
        reward_episode_1 = [reward_ratio] * len(observation_episode_1)
        reward_episode_2 = [reward_ratio] * len(observation_episode_2)
        reward_episode_3 = [reward_ratio] * len(observation_episode_3)

        RL_1.store_training_samples_per_episode(observation_episode_1, action_episode_1, reward_episode_1)
        RL_2.store_training_samples_per_episode(observation_episode_2, action_episode_2, reward_episode_2)
        RL_3.store_training_samples_per_episode(observation_episode_3, action_episode_3, reward_episode_3)

        """
        check_tput_quality(tput)
        """
        if not use_external_knowledge:
            if highest_tput < tput:
                highest_tput_original = highest_tput
                optimal_range_original = optimal_range
                highest_tput = tput
                highest_value_time = time.time() - start_time

                observation_optimal_1, action_optimal_1, reward_optimal_1 = [], [], []
                observation_optimal_2, action_optimal_2, reward_optimal_2 = [], [], []
                observation_optimal_3, action_optimal_3, reward_optimal_3 = [], [], []

                observation_optimal_1.extend(observation_episode_1)
                action_optimal_1.extend(action_episode_1)
                reward_optimal_1.extend(reward_episode_1)

                observation_optimal_2.extend(observation_episode_2)
                action_optimal_2.extend(action_episode_2)
                reward_optimal_2.extend(reward_episode_2)

                observation_optimal_3.extend(observation_episode_3)
                action_optimal_3.extend(action_episode_3)
                reward_optimal_3.extend(reward_episode_3)
                allocation_optimal = env.state

                optimal_range = min(1.02, highest_tput / (highest_tput_original / optimal_range_original))
            elif highest_tput < tput * optimal_range:
                observation_optimal_1.extend(observation_episode_1)
                action_optimal_1.extend(action_episode_1)
                reward_optimal_1.extend(reward_episode_1)

                observation_optimal_2.extend(observation_episode_2)
                action_optimal_2.extend(action_episode_2)
                reward_optimal_2.extend(reward_episode_2)

                observation_optimal_3.extend(observation_episode_3)
                action_optimal_3.extend(action_episode_3)
                reward_optimal_3.extend(reward_episode_3)
        else:
            if highest_demo_tput < tput:
                highest_demo_tput = tput
                allocation_demo = env.state
            observation_demo_1.extend(observation_episode_1)
            action_demo_1.extend(action_episode_1)
            reward_demo_1.extend(reward_episode_1)

            observation_demo_2.extend(observation_episode_2)
            action_demo_2.extend(action_episode_2)
            reward_demo_2.extend(reward_episode_2)

            observation_demo_3.extend(observation_episode_3)
            action_demo_3.extend(action_episode_3)
            reward_demo_3.extend(reward_episode_3)

        observation_episode_1, action_episode_1, reward_episode_1 = [], [], []
        observation_episode_2, action_episode_2, reward_episode_2 = [], [], []
        observation_episode_3, action_episode_3, reward_episode_3 = [], [], []


        """
        Each batch, RL.learn()
        """
        records_per_episode = NUM_CONTAINERS * training_times_per_episode
        if (epoch_i % batch_size == 0) & (epoch_i > 1):
            buffer_size = int(len(reward_optimal_1) / records_per_episode)

            if buffer_size < replay_size:
                # TODO: if layers changes, training_times_per_episode should be modified
                RL_1.ep_obs.extend(observation_optimal_1)
                RL_1.ep_as.extend(action_optimal_1)
                RL_1.ep_rs.extend(reward_optimal_1)

                RL_2.ep_obs.extend(observation_optimal_2)
                RL_2.ep_as.extend(action_optimal_2)
                RL_2.ep_rs.extend(reward_optimal_2)

                RL_3.ep_obs.extend(observation_optimal_3)
                RL_3.ep_as.extend(action_optimal_3)
                RL_3.ep_rs.extend(reward_optimal_3)

            else:
                replay_index = np.random.choice(range(buffer_size), size= replay_size, replace=False)
                for replay_id in range(replay_size):
                    replace_start = replay_index[replay_id]
                    RL_1.ep_obs.extend(observation_optimal_1[replace_start * int(records_per_episode): (replace_start+1) * int(records_per_episode)])
                    RL_1.ep_as.extend(action_optimal_1[replace_start * int(records_per_episode): (replace_start+1) * int(records_per_episode)])
                    RL_1.ep_rs.extend(reward_optimal_1[replace_start * int(records_per_episode): (replace_start+1) * int(records_per_episode)])

                    RL_2.ep_obs.extend(observation_optimal_2[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                    RL_2.ep_as.extend(action_optimal_2[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                    RL_2.ep_rs.extend(reward_optimal_2[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])

                    RL_3.ep_obs.extend(observation_optimal_3[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                    RL_3.ep_as.extend(action_optimal_3[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                    RL_3.ep_rs.extend(reward_optimal_3[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])



            demo_size = int(len(reward_demo_1) / records_per_episode)
            demo_replay_size = 3
            if demo_size>0:
                replay_demo_index = np.random.choice(range(demo_size), size=demo_replay_size, replace=False)
                if (np.random.rand() > -1) & (highest_demo_tput > highest_tput * 0.95):
                    for replay_id in range(demo_replay_size):
                        replace_start = replay_demo_index[replay_id]
                        # replace_start = replay_demo_index[0]
                        RL_1.ep_obs.extend(observation_demo_1[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                        RL_1.ep_as.extend(action_demo_1[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                        RL_1.ep_rs.extend(reward_demo_1[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])

                        RL_2.ep_obs.extend(observation_demo_2[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                        RL_2.ep_as.extend(action_demo_2[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                        RL_2.ep_rs.extend(reward_demo_2[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])

                        RL_3.ep_obs.extend(observation_demo_3[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                        RL_3.ep_as.extend(action_demo_3[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])
                        RL_3.ep_rs.extend(reward_demo_3[replace_start * int(records_per_episode): (replace_start + 1) * int(records_per_episode)])

            RL_1.learn(epoch_i, entropy_weight, True)
            RL_2.learn(epoch_i, entropy_weight, False)
            RL_3.learn(epoch_i, entropy_weight, False)
            print("time duration: %f" % (time.time() - start_time))

        """
        checkpoint, per 1000 episodes
        """
        if (epoch_i % 100 == 0) & (epoch_i > 1):
            count_size = (len(reward_optimal_1) / (NUM_CONTAINERS * training_times_per_episode))
            print("\n epoch: %d, highest tput: %f, times: %d" % (epoch_i, highest_tput, count_size))
            print("spending time: %f"%(time.time() - global_start_time))
            RL_1.save_session(ckpt_path)
            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode), time=np.array(RL_1.time_persisit),
                     highest_value_time=highest_value_time, highest_tput=highest_tput,
                     allocation_optimal=allocation_optimal,allocation_demo=allocation_demo)

            """
            optimal range adaptively change
            """
            if (len(reward_optimal_1) / records_per_episode) > replay_size * 2:
                optimal_range *= 0.99
                optimal_range = max(optimal_range, 1.001)
                # 81: save the demonstration
                observation_optimal_original_1 = copy.deepcopy(observation_optimal_1)
                action_optimal_original_1 = copy.deepcopy(action_optimal_1)
                reward_optimal_original_1 = copy.deepcopy(reward_optimal_1)

                observation_optimal_1, action_optimal_1, reward_optimal_1 = [], [], []

                observation_optimal_1.extend(observation_optimal_original_1[:int(records_per_episode)])
                action_optimal_1.extend(action_optimal_original_1[:int(records_per_episode)])
                reward_optimal_1.extend(reward_optimal_original_1[:int(records_per_episode)])

                observation_optimal_1.extend(observation_optimal_original_1[int(len(observation_optimal_original_1) - replay_size * records_per_episode):])
                action_optimal_1.extend(action_optimal_original_1[int(len(action_optimal_original_1) - replay_size * records_per_episode):])
                reward_optimal_1.extend(reward_optimal_original_1[int(len(reward_optimal_original_1) - replay_size * records_per_episode):])

                observation_optimal_original_2 = copy.deepcopy(observation_optimal_2)
                action_optimal_original_2 = copy.deepcopy(action_optimal_2)
                reward_optimal_original_2 = copy.deepcopy(reward_optimal_2)

                observation_optimal_2, action_optimal_2, reward_optimal_2 = [], [], []

                observation_optimal_2.extend(observation_optimal_original_2[:int(records_per_episode)])
                action_optimal_2.extend(action_optimal_original_2[:int(records_per_episode)])
                reward_optimal_2.extend(reward_optimal_original_2[:int(records_per_episode)])

                observation_optimal_2.extend(observation_optimal_original_2[int(len(observation_optimal_original_2) - replay_size * records_per_episode):])
                action_optimal_2.extend(action_optimal_original_2[int(len(action_optimal_original_2) - replay_size * records_per_episode):])
                reward_optimal_2.extend(reward_optimal_original_2[int(len(reward_optimal_original_2) - replay_size * records_per_episode):])

                observation_optimal_original_3 = copy.deepcopy(observation_optimal_3)
                action_optimal_original_3 = copy.deepcopy(action_optimal_3)
                reward_optimal_original_3 = copy.deepcopy(reward_optimal_3)

                observation_optimal_3, action_optimal_3, reward_optimal_3 = [], [], []

                observation_optimal_3.extend(observation_optimal_original_3[:int(records_per_episode)])
                action_optimal_3.extend(action_optimal_original_3[:int(records_per_episode)])
                reward_optimal_3.extend(reward_optimal_original_3[:int(records_per_episode)])

                observation_optimal_3.extend(observation_optimal_original_3[int(len(observation_optimal_original_3) - replay_size * records_per_episode):])
                action_optimal_3.extend(action_optimal_original_3[int(len(action_optimal_original_3) - replay_size * records_per_episode):])
                reward_optimal_3.extend(reward_optimal_original_3[int(len(reward_optimal_original_3) - replay_size * records_per_episode):])

        epoch_i += 1
        if epoch_i > 30:
            use_external_knowledge = False


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
    params['path'] = "729nodes_" + str(hyper_parameter['batch_set_choice']) + "_" + str(hyper_parameter['batch_C_numbers'])
    params['number of containers'] = hyper_parameter['batch_C_numbers']

    make_path(params['path'])

    train(params)


if __name__ == "__main__":
    main()
