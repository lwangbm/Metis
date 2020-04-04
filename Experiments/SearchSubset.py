import numpy as np
import time
import os
import sys
sys.path.append("/Users/ourokutaira/Desktop/George")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbed.cluster_env import LraClusterEnv
from testbed.PolicyGradient_RL2 import PolicyGradient
import argparse
from testbed.simulator.simulator import Simulator

"""
'--batch_choice': 0, 1, 2, ``` 30
python3 SearchSubset.py --batch_choice 0
"""
hyper_parameter = {
        'batch_C_numbers': None
}
params = {
        'batch_size': 50,
        'epochs': 60000,
        'path': "cpo_compare_RL2_" + str(hyper_parameter['batch_C_numbers']),
        'recover': False,
        'learning rate': 0.01,
        'nodes per group': 3,
        'number of nodes in the cluster': 27,
        'replay size': 50,
        'container_limitation per node': 8
}
NUM_CONTAINERS = 100
sim = Simulator()


def handle_constraint(observation, NUM_NODES):

    observation_original = observation.copy()
    mapping_index = []
    # TODO: we could add more constraints here
    list_check_1 = observation[:, :].sum(1) > params['container_limitation per node']   # >8
    list_check_2 = (observation > 1).any(1)
    list_check_3 = ((observation[:, 1] > 0) * (observation[:, 2] > 0))>0
    list_check = list_check_1 | list_check_2 | list_check_3

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
    env = LraClusterEnv(num_nodes=NUM_NODES)
    batch_size = params['batch_size']
    ckpt_path_1 = "./checkpoint/" + params['path'] + "1/model.ckpt"
    ckpt_path_2 = "./checkpoint/" + params['path'] + "2/model.ckpt"
    ckpt_path_3 = "./checkpoint/" + params['path'] + "3/model.ckpt"
    make_path(params['path'] + "1")
    make_path(params['path'] + "2")
    make_path(params['path'] + "3")
    useExternal = False
    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    Recover = params['recover']
    nodes_per_group = int(params['nodes per group'])
    replay_size = params['replay size']
    training_times_per_episode = 1  # TODO: if layers changes, training_times_per_episode should be modified
    """
    Build Network
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * (env.NUM_APPS + 1 + env.NUM_APPS )+ 1 + env.NUM_APPS)  #: 3*9+1 = 28
    RL_1 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix="100" + '1a')

    RL_2 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix="100" + '2a')

    RL_3 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix="100" + '3a')

    """
    Training
    """
    start_time = time.time()
    global_start_time = start_time
    number_optimal = []
    observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1 = [], [], [], []
    observation_optimal_1, action_optimal_1, reward_optimal_1, safety_optimal_1 = [], [], [], []
    observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2 = [], [], [], []
    observation_optimal_2, action_optimal_2, reward_optimal_2, safety_optimal_2 = [], [], [], []
    observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3 = [], [], [], []
    observation_optimal_3, action_optimal_3, reward_optimal_3, safety_optimal_3 = [], [], [], []

    epoch_i = 0
    thre_entropy = 0.1

    names = locals()
    for i in range(0, 12):
        names['highest_tput_' + str(i)] = 0
        names['observation_optimal_1_' + str(i)] = []
        names['action_optimal_1_' + str(i)] = []
        names['observation_optimal_2_' + str(i)] = []
        names['action_optimal_2_' + str(i)] = []
        names['observation_optimal_3_' + str(i)] = []
        names['action_optimal_3_' + str(i)] = []
        names['reward_optimal_1_' + str(i)] = []
        names['reward_optimal_2_' + str(i)] = []
        names['reward_optimal_3_' + str(i)] = []
        names['safety_optimal_1_' + str(i)] = []
        names['safety_optimal_2_' + str(i)] = []
        names['safety_optimal_3_' + str(i)] = []
        names['number_optimal_' + str(i)] = []
        names['optimal_range_' + str(i)] = 1.05
        names['lowest_vio_' + str(i)] = 500
        names['observation_optimal_1_vio_' + str(i)] = []
        names['action_optimal_1_vio_' + str(i)] = []
        names['observation_optimal_2_vio_' + str(i)] = []
        names['action_optimal_2_vio_' + str(i)] = []
        names['observation_optimal_3_vio_' + str(i)] = []
        names['action_optimal_3_vio_' + str(i)] = []
        names['reward_optimal_vio_1_' + str(i)] = []
        names['reward_optimal_vio_2_' + str(i)] = []
        names['reward_optimal_vio_3_' + str(i)] = []
        names['safety_optimal_vio_1_' + str(i)] = []
        names['safety_optimal_vio_2_' + str(i)] = []
        names['safety_optimal_vio_3_' + str(i)] = []
        names['number_optimal_vio_' + str(i)] = []
        names['optimal_range_vio_' + str(i)] = 1.1

    def store_episode_1(observations, actions):
        observation_episode_1.append(observations)
        action_episode_1.append(actions)

    def store_episode_2(observations, actions):
        observation_episode_2.append(observations)
        action_episode_2.append(actions)

    def store_episode_3(observations, actions):
        observation_episode_3.append(observations)
        action_episode_3.append(actions)

    tput_origimal_class = 0
    source_batch_, index_data_ = batch_data(NUM_CONTAINERS, env.NUM_APPS)  # index_data = [0,1,2,0,1,2]

    break_number = 0
    while epoch_i < params['epochs']:
        break_flag = False
        observation = env.reset().copy()  # (9,9)
        source_batch = source_batch_.copy()
        index_data = index_data_.copy()

        """
        Episode
        """
        """
        first layer
        """
        source_batch_first = source_batch_.copy()
        observation_first_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
        for inter_episode_index in range(NUM_CONTAINERS):

            appid = index_data[inter_episode_index]
            source_batch_first[appid] -= 1
            observation_first_layer_copy = observation_first_layer.copy()
            observation_first_layer_copy[:, appid] += 1
            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy > 9 * 2, axis=1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
            observation_first_layer_copy = np.array(observation_first_layer_copy).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, appid).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, np.array(source_batch_first)).reshape(1, -1)
            if useExternal:
                action_1 = inter_episode_index%3
                prob_weights = []
            else:
                action_1, prob_weights = RL_1.choose_action(observation_first_layer_copy.copy())
            observation_first_layer[action_1, appid] += 1
            store_episode_1(observation_first_layer_copy, action_1)

        """
        second layer
        """
        observation_second_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20

        number_cont_second_layer = []

        for second_layer_index in range(nodes_per_group):

            rnd_array = observation_first_layer[second_layer_index].copy()
            source_batch_second, index_data = batch_data_sub(rnd_array)
            observation_second_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
            NUM_CONTAINERS_second = sum(source_batch_second)
            number_cont_second_layer.append(NUM_CONTAINERS_second)

            for inter_episode_index in range(NUM_CONTAINERS_second):

                appid = index_data[inter_episode_index]
                source_batch_second[appid] -= 1
                observation_second_layer_copy = observation_second_layer.copy()
                observation_second_layer_copy[:, appid] += 1
                observation_second_layer_copy = np.append(observation_second_layer_copy, observation_second_layer_copy > 3 * 2, axis=1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, observation_second_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                observation_second_layer_copy = np.array(observation_second_layer_copy).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, appid).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, np.array(source_batch_second)).reshape(1, -1)
                if useExternal:
                    action_2 = inter_episode_index % 3
                    prob_weights = []
                else:
                    action_2, prob_weights = RL_2.choose_action(observation_second_layer_copy.copy())
                observation_second_layer[action_2, appid] += 1
                store_episode_2(observation_second_layer_copy, action_2)
            observation_second_layer_aggregation = np.append(observation_second_layer_aggregation, observation_second_layer, 0)

        """
        third layer
        """
        observation_third_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20
        number_cont_third_layer = []

        for third_layer_index in range(nodes_per_group * nodes_per_group):

            rnd_array = observation_second_layer_aggregation[third_layer_index].copy()
            source_batch_third, index_data = batch_data_sub(rnd_array)
            observation_third_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
            NUM_CONTAINERS_third = sum(source_batch_third)
            number_cont_third_layer.append(NUM_CONTAINERS_third)

            for inter_episode_index in range(NUM_CONTAINERS_third):
                appid = index_data[inter_episode_index]
                source_batch_third[appid] -= 1
                observation_third_layer_copy = observation_third_layer.copy()
                observation_third_layer_copy[:, appid] += 1
                observation_third_layer_copy, mapping_index = handle_constraint(observation_third_layer_copy.copy(), 3)
                if len(mapping_index) < 1:
                    break_flag = True
                    break
                assert len(mapping_index) > 0
                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy > 1 * 2, axis=1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                observation_third_layer_copy = np.array(observation_third_layer_copy).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, appid).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, np.array(source_batch_third)).reshape(1, -1)
                if useExternal:
                    action_3 = inter_episode_index % 3
                    prob_weights = []
                else:
                    action_3, prob_weights = RL_3.choose_action(observation_third_layer_copy.copy())
                observation_third_layer[mapping_index[action_3], appid] += 1
                store_episode_3(observation_third_layer_copy, action_3)
            if break_flag:
                break
            observation_third_layer_aggregation = np.append(observation_third_layer_aggregation, observation_third_layer, 0)
        if break_flag:
            break_number += 1

        """
        After an entire allocation, calculate total throughput, reward
        """
        if not break_flag:
            env.state = observation_third_layer_aggregation.copy()
            tput_state = env.get_tput_total_env()
            tput = (sim.predict(tput_state.reshape(-1, env.NUM_APPS)) * tput_state).sum() / NUM_CONTAINERS
            assert sum(sum(env.state)) == NUM_CONTAINERS
            assert (env.state.sum(0) == source_batch_).all()
            list_check = 0
            for node in range(NUM_NODES):
                for app in range(env.NUM_APPS):
                    if env.state[node, :].sum() > params['container_limitation per node'] or env.state[node, app] > 1 or (app == 1 and env.state[node, 2] > 0) or (app == 2 and env.state[node, 1] > 0):
                        list_check += env.state[node, app]
            list_check_ratio = -1.0 * list_check / NUM_CONTAINERS
        else:
            tput = 0
            list_check_ratio = 0
            list_check, list_check_per_app, list_check_coex, list_check_sum = 0,0,0, 0

        safety_episode_3 = [list_check_ratio * 1.0] * len(observation_episode_3)
        reward_episode_3 = [tput * 1.0] * len(observation_episode_3)

        safety_episode_2, reward_episode_2  = [], []
        for second_subcluster_index in range(nodes_per_group):
            safety_episode_2.extend([list_check_ratio * 1.0] * int(number_cont_second_layer[second_subcluster_index]))
            reward_episode_2.extend([tput * 1.0] * int(number_cont_second_layer[second_subcluster_index]))

        safety_episode_1 = [list_check_ratio * 1.0] * len(observation_episode_1)
        reward_episode_1 = [tput * 1.0] * len(observation_episode_1)

        RL_1.store_tput_per_episode(tput, list_check, epoch_i, [], [], list_check)
        RL_2.store_tput_per_episode(tput, list_check, epoch_i, [],[],[])
        RL_3.store_tput_per_episode(tput, list_check, epoch_i, [],[],[])
        RL_1.store_training_samples_per_episode(observation_episode_1, action_episode_1, safety_episode_1, reward_episode_1)
        RL_2.store_training_samples_per_episode(observation_episode_2, action_episode_2, safety_episode_2, reward_episode_2)
        RL_3.store_training_samples_per_episode(observation_episode_3, action_episode_3, safety_episode_3, reward_episode_3)

        """
        check_tput_quality(tput)
        """
        if names['highest_tput_' + str(tput_origimal_class)] < tput and list_check_ratio==0:
            names['highest_tput_' + str(tput_origimal_class)] = tput
            names['observation_optimal_1_' + str(tput_origimal_class)], names['action_optimal_1_' + str(tput_origimal_class)], names['observation_optimal_2_' + str(tput_origimal_class)], names[
                'action_optimal_2_' + str(tput_origimal_class)], names['reward_optimal_1_' + str(tput_origimal_class)], names['reward_optimal_2_' + str(tput_origimal_class)], names[
                'reward_optimal_3_' + str(tput_origimal_class)], names['number_optimal_' + str(tput_origimal_class)], names['safety_optimal_1_' + str(tput_origimal_class)], names[
                'safety_optimal_2_' + str(tput_origimal_class)], names['safety_optimal_3_' + str(tput_origimal_class)] = [], [], [], [], [], [], [], [], [], [], []
            names['observation_optimal_3_' + str(tput_origimal_class)], names['action_optimal_3_' + str(tput_origimal_class)] = [], []
            names['observation_optimal_1_' + str(tput_origimal_class)].extend(observation_episode_1)
            names['action_optimal_1_' + str(tput_origimal_class)].extend(action_episode_1)
            names['observation_optimal_2_' + str(tput_origimal_class)].extend(observation_episode_2)
            names['action_optimal_2_' + str(tput_origimal_class)].extend(action_episode_2)
            names['observation_optimal_3_' + str(tput_origimal_class)].extend(observation_episode_3)
            names['action_optimal_3_' + str(tput_origimal_class)].extend(action_episode_3)
            names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)
            names['safety_optimal_1_' + str(tput_origimal_class)].extend(safety_episode_1)
            names['safety_optimal_2_' + str(tput_origimal_class)].extend(safety_episode_2)
            names['safety_optimal_3_' + str(tput_origimal_class)].extend(safety_episode_3)
            names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)
            names['reward_optimal_2_' + str(tput_origimal_class)].extend(reward_episode_2)
            names['reward_optimal_3_' + str(tput_origimal_class)].extend(reward_episode_3)
            names['optimal_range_' + str(tput_origimal_class)] = 1.05
        observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1, reward_episode_1 = [], [], [], [], []
        observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2, reward_episode_2 = [], [], [], [], []
        observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3, reward_episode_3 = [], [], [], [], []

        """
        Each batch, RL.learn()
        """
        if (epoch_i % batch_size == 0) & (epoch_i > batch_size+1):
            for replay_class in range(0, 1):

                number_optimal = names['number_optimal_' + str(replay_class)]

                reward_optimal_1 = names['reward_optimal_1_' + str(replay_class)]
                reward_optimal_2 = names['reward_optimal_2_' + str(replay_class)]
                reward_optimal_3 = names['reward_optimal_3_' + str(replay_class)]
                safety_optimal_1 = names['safety_optimal_1_' + str(replay_class)]
                safety_optimal_2 = names['safety_optimal_2_' + str(replay_class)]
                safety_optimal_3 = names['safety_optimal_3_' + str(replay_class)]

                observation_optimal_1 = names['observation_optimal_1_' + str(replay_class)]
                action_optimal_1 = names['action_optimal_1_' + str(replay_class)]
                observation_optimal_2 = names['observation_optimal_2_' + str(replay_class)]
                action_optimal_2 = names['action_optimal_2_' + str(replay_class)]
                observation_optimal_3 = names['observation_optimal_3_' + str(replay_class)]
                action_optimal_3 = names['action_optimal_3_' + str(replay_class)]

                buffer_size = int(len(number_optimal))

                if buffer_size < replay_size:
                    # TODO: if layers changes, training_times_per_episode should be modified
                    RL_1.ep_obs.extend(observation_optimal_1)
                    RL_1.ep_as.extend(action_optimal_1)
                    RL_1.ep_rs.extend(reward_optimal_1)
                    RL_1.ep_ss.extend(safety_optimal_1)

                    RL_2.ep_obs.extend(observation_optimal_2)
                    RL_2.ep_as.extend(action_optimal_2)
                    RL_2.ep_rs.extend(reward_optimal_2)
                    RL_2.ep_ss.extend(safety_optimal_2)

                    RL_3.ep_obs.extend(observation_optimal_3)
                    RL_3.ep_as.extend(action_optimal_3)
                    RL_3.ep_rs.extend(reward_optimal_3)
                    RL_3.ep_ss.extend(safety_optimal_3)

                else:
                    replay_index = np.random.choice(range(buffer_size), size=replay_size, replace=False)
                    for replay_id in range(replay_size):
                        replace_start = replay_index[replay_id]
                        start_location = sum(number_optimal[:replace_start])
                        stop_location = sum(number_optimal[:replace_start + 1])
                        RL_1.ep_obs.extend(observation_optimal_1[start_location: stop_location])
                        RL_1.ep_as.extend(action_optimal_1[start_location: stop_location])
                        RL_1.ep_rs.extend(reward_optimal_1[start_location: stop_location])
                        RL_1.ep_ss.extend(safety_optimal_1[start_location: stop_location])

                        RL_2.ep_obs.extend(observation_optimal_2[start_location: stop_location])
                        RL_2.ep_as.extend(action_optimal_2[start_location: stop_location])
                        RL_2.ep_rs.extend(reward_optimal_2[start_location: stop_location])
                        RL_2.ep_ss.extend(safety_optimal_2[start_location: stop_location])

                        RL_3.ep_obs.extend(observation_optimal_3[start_location: stop_location])
                        RL_3.ep_as.extend(action_optimal_3[start_location: stop_location])
                        RL_3.ep_rs.extend(reward_optimal_3[start_location: stop_location])
                        RL_3.ep_ss.extend(safety_optimal_3[start_location: stop_location])

            RL_1.learn(epoch_i, thre_entropy, IfPrint=True)
            RL_2.learn(epoch_i, thre_entropy)
            if len(RL_3.ep_obs)>1:
                RL_3.learn(epoch_i, thre_entropy)

        """
        checkpoint, per 1000 episodes
        """
        if (epoch_i % 3000 == 0) & (epoch_i > 1):
            RL_1.save_session(ckpt_path_1)
            RL_2.save_session(ckpt_path_2)
            RL_3.save_session(ckpt_path_3)
            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode), vi_perapp=np.array(RL_1.ss_perapp_persisit), vi_coex=np.array(RL_1.ss_coex_persisit), vi_sum=np.array(RL_1.ss_sum_persisit), break_number=break_number)
            """
            optimal range adaptively change
            """
            thre_entropy *= 0.5
            thre_entropy = max(thre_entropy, 0.01)
        epoch_i += 1


def batch_data(NUM_CONTAINERS, NUM_NODES):
    npzfile = np.load("./data/batch_set_cpo_27node_" + str(100) + '.npz')
    batch_set = npzfile['batch_set']
    rnd_array = batch_set[hyper_parameter['batch_C_numbers'], :]
    index_data = []
    for i in range(7):
        index_data.extend([i] * rnd_array[i])
    print(hyper_parameter['batch_C_numbers'])
    print(rnd_array)
    return rnd_array, index_data


def batch_data_sub(rnd_array):
    rnd_array = rnd_array.copy()
    index_data = []
    for i in range(7):
        index_data.extend([i] * int(rnd_array[i]))
    return rnd_array, index_data


def make_path(dirname):
    if not os.path.exists("./checkpoint/" + dirname):
        os.mkdir("./checkpoint/"+ dirname)
        print("Directory ", "./checkpoint/" + dirname, " Created ")
    else:
        print("Directory ", "./checkpoint/" + dirname, " already exists")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_choice', type=int)
    args = parser.parse_args()
    hyper_parameter['batch_C_numbers'] = args.batch_choice
    params['path'] = "cpo_compare_RL1_" + str(hyper_parameter['batch_C_numbers'])
    make_path(params['path'])
    train(params)

if __name__ == "__main__":
    main()
