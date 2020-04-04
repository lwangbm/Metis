import numpy as np
import time
import os
import sys

sys.path.append("/Users/ourokutaira/Desktop/George")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbed.cluster_env import LraClusterEnv
from testbed.PolicyGradient_CPO import PolicyGradient
import argparse
from testbed.simulator.simulator import Simulator

"""
'--start_sample: 0, 10, 20, ``` 210
python3 TrainSubSchedulerWith27Nodes.py --start_sample 0
"""

params = {'batch_size': 500, 'epochs': 8000000, 'path': "cpo_separate_level_sc_", 'rec_path': "cpo_separate_level_sc_", 'recover': False, 'learning rate': 0.005, 'nodes per group': 3,
    'number of nodes in the cluster': 27, 'replay size': 10, 'container_limitation per node': 8}


def train(params):
    """
    parameters set
    """
    NUM_NODES = params['number of nodes in the cluster']
    NUM_CONTAINERS_start = params['NUM_CONTAINERS_start']
    env = LraClusterEnv(num_nodes=NUM_NODES)
    batch_size = params['batch_size']
    ckpt_path_1 = "./checkpoint/" + params['path'] + "1/model.ckpt"
    ckpt_path_2 = "./checkpoint/" + params['path'] + "2/model.ckpt"
    ckpt_path_3 = "./checkpoint/" + params['path'] + "3/model.ckpt"
    ckpt_path_recover_1 = "./checkpoint/" + params['rec_path'] + "1/model.ckpt"
    ckpt_path_recover_2 = "./checkpoint/" + params['rec_path'] + "2/model.ckpt"
    ckpt_path_recover_3 = "./checkpoint/" + params['rec_path'] + "3/model.ckpt"
    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    Recover = params['recover']
    nodes_per_group = int(params['nodes per group'])
    replay_size = params['replay size']
    training_times_per_episode = 1  # TODO: if layers changes, training_times_per_episode should be modified
    safety_requirement = 2 / 100.

    """
    Build Network
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * (env.NUM_APPS + 1 + env.NUM_APPS) + 1 + env.NUM_APPS)  #: 3*9+1 = 28
    RL_1 = PolicyGradient(n_actions=n_actions, n_features=n_features, learning_rate=params['learning rate'], suffix=str(params['NUM_CONTAINERS_start']) + '1a', safety_requirement=safety_requirement)

    RL_2 = PolicyGradient(n_actions=n_actions, n_features=n_features, learning_rate=params['learning rate'], suffix=str(params['NUM_CONTAINERS_start']) + '2a', safety_requirement=safety_requirement)

    RL_3 = PolicyGradient(n_actions=n_actions, n_features=n_features, learning_rate=params['learning rate'], suffix=str(params['NUM_CONTAINERS_start']) + '3a', safety_requirement=safety_requirement)

    sim = Simulator()

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
    thre_entropy = 0.002

    names = locals()
    for i in range(0, 10):
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

    def store_episode_1(observations, actions):
        observation_episode_1.append(observations)
        action_episode_1.append(actions)

    def store_episode_2(observations, actions):
        observation_episode_2.append(observations)
        action_episode_2.append(actions)

    def store_episode_3(observations, actions):
        observation_episode_3.append(observations)
        action_episode_3.append(actions)

    while epoch_i < params['epochs']:
        if Recover:
            RL_1.restore_session(ckpt_path_recover_1)
            RL_2.restore_session(ckpt_path_recover_2)
            RL_3.restore_session(ckpt_path_recover_3)
            Recover = False

        NUM_CONTAINERS = np.random.randint(NUM_CONTAINERS_start + 1, NUM_CONTAINERS_start + 11)
        tput_origimal_class = int(NUM_CONTAINERS - NUM_CONTAINERS_start - 1)
        source_batch_, index_data = batch_data(NUM_CONTAINERS, env.NUM_APPS)  # index_data = [0,1,2,0,1,2]
        observation = env.reset().copy()  # (9,9)
        source_batch = source_batch_.copy()

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
                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy > 1 * 2, axis=1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                observation_third_layer_copy = np.array(observation_third_layer_copy).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, appid).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, np.array(source_batch_third)).reshape(1, -1)
                action_3, prob_weights = RL_3.choose_action(observation_third_layer_copy.copy())
                observation_third_layer[action_3, appid] += 1
                store_episode_3(observation_third_layer_copy, action_3)

            observation_third_layer_aggregation = np.append(observation_third_layer_aggregation, observation_third_layer, 0)

        """
        After an entire allocation, calculate total throughput, reward
        """
        env.state = observation_third_layer_aggregation.copy()
        assert sum(sum(env.state)) == NUM_CONTAINERS
        assert (env.state.sum(0) == source_batch_).all()
        tput_state = env.state
        tput_breakdown = sim.predict(tput_state.reshape(-1, env.NUM_APPS))
        tput = (tput_breakdown * tput_state).sum() / NUM_CONTAINERS
        reward_ratio = (tput - 0)

        state = env.state
        # list_check_per_app = (env.state > 1).sum()  # + max((env.state - 1).max(), 0)
        # list_check_sum = sum(env.state.sum(1) > params['container_limitation per node'])  # + max(max(env.state.sum(1) - params['container_limitation per node']), 0)
        # list_check_coex = sum((env.state[:, 1] > 0) * (env.state[:, 2] > 0))
        # list_check = list_check_sum + list_check_coex + list_check_per_app
        list_check = 0
        for node in range(NUM_NODES):
            for app in range(env.NUM_APPS):
                if env.state[node, :].sum() > params['container_limitation per node'] or env.state[node, app] > 1 or (app == 1 and env.state[node, 2] > 0) or (app == 2 and env.state[node, 1] > 0):
                    list_check += env.state[node, app]

        list_check_ratio = list_check / NUM_CONTAINERS

        safety_episode_1 = [list_check_ratio * 1.0] * len(observation_episode_1)
        reward_episode_1 = [reward_ratio * 1.0] * len(observation_episode_1)
        safety_episode_2 = [list_check_ratio * 1.0] * len(observation_episode_2)
        reward_episode_2 = [reward_ratio * 1.0] * len(observation_episode_2)
        safety_episode_3 = [list_check_ratio * 1.0] * len(observation_episode_3)
        reward_episode_3 = [reward_ratio * 1.0] * len(observation_episode_3)

        RL_1.store_tput_per_episode(tput, epoch_i, list_check, list_check_per_app, list_check_coex, list_check_sum)
        RL_2.store_tput_per_episode(tput, epoch_i, list_check, list_check_per_app, list_check_coex, list_check_sum)
        RL_3.store_tput_per_episode(tput, epoch_i, list_check, list_check_per_app, list_check_coex, list_check_sum)
        RL_1.store_training_samples_per_episode(observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1)
        RL_2.store_training_samples_per_episode(observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2)
        RL_3.store_training_samples_per_episode(observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3)

        """
        check_tput_quality(tput)
        """
        if list_check <= safety_requirement * 0.5:
            if names['highest_tput_' + str(tput_origimal_class)] < tput:
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

            elif names['highest_tput_' + str(tput_origimal_class)] < tput * names['optimal_range_' + str(tput_origimal_class)]:
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

        observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1 = [], [], [], []
        observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2 = [], [], [], []
        observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3 = [], [], [], []

        """
        Each batch, RL.learn()
        """
        if (epoch_i % batch_size == 0) & (epoch_i > 1):
            for replay_class in range(0, 10):

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

            RL_1.learn(epoch_i, thre_entropy, Ifprint=True)
            RL_2.learn(epoch_i, thre_entropy)
            optim_case = RL_3.learn(epoch_i, thre_entropy)

        """
        checkpoint, per 1000 episodes
        """
        if (epoch_i % 5000 == 0) & (epoch_i > 1):

            RL_1.save_session(ckpt_path_1)
            RL_2.save_session(ckpt_path_2)
            RL_3.save_session(ckpt_path_3)
            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode), vi_coex=np.array(RL_1.coex_persisit), vio_sum=np.array(RL_1.sum_persisit))
            """
            optimal range adaptively change
            """
            for class_replay in range(0, 10):
                number_optimal = names['number_optimal_' + str(class_replay)]
                count_size = int(len(number_optimal))

                if (count_size > 100):
                    names['optimal_range_' + str(class_replay)] *= 0.99
                    names['optimal_range_' + str(class_replay)] = max(names['optimal_range_' + str(class_replay)], 1.01)
                    start_location = sum(names['number_optimal_' + str(class_replay)][:-10]) * training_times_per_episode
                    names['observation_optimal_1_' + str(class_replay)] = names['observation_optimal_1_' + str(class_replay)][start_location:]
                    names['action_optimal_1_' + str(class_replay)] = names['action_optimal_1_' + str(class_replay)][start_location:]
                    names['observation_optimal_2_' + str(class_replay)] = names['observation_optimal_2_' + str(class_replay)][start_location:]
                    names['action_optimal_2_' + str(class_replay)] = names['action_optimal_2_' + str(class_replay)][start_location:]
                    names['observation_optimal_3_' + str(class_replay)] = names['observation_optimal_3_' + str(class_replay)][start_location:]
                    names['action_optimal_3_' + str(class_replay)] = names['action_optimal_3_' + str(class_replay)][start_location:]
                    names['number_optimal_' + str(class_replay)] = names['number_optimal_' + str(class_replay)][-10:]
                    names['safety_optimal_1_' + str(class_replay)] = names['safety_optimal_1_' + str(class_replay)][start_location:]
                    names['safety_optimal_2_' + str(class_replay)] = names['safety_optimal_2_' + str(class_replay)][start_location:]
                    names['safety_optimal_3_' + str(class_replay)] = names['safety_optimal_3_' + str(class_replay)][start_location:]
                    names['reward_optimal_1_' + str(class_replay)] = names['reward_optimal_1_' + str(class_replay)][start_location:]
                    names['reward_optimal_2_' + str(class_replay)] = names['reward_optimal_2_' + str(class_replay)][start_location:]
                    names['reward_optimal_3_' + str(class_replay)] = names['reward_optimal_3_' + str(class_replay)][start_location:]

                print("optimal_range:", names['optimal_range_' + str(class_replay)])

            if optim_case > 0:
                thre_entropy *= 0.5
            thre_entropy = max(thre_entropy, 0.0001)

        epoch_i += 1


def batch_data(NUM_CONTAINERS, NUM_NODES):
    _sum = NUM_CONTAINERS
    n = NUM_NODES
    while True:
        rnd_array = np.random.multinomial(_sum, np.ones(n) / n, size=1)[0]
        if (rnd_array <= 27 * 1).all() and (rnd_array[1] <= 10) and (rnd_array[2] <= 10):
            break
    index_data = []
    for i in range(NUM_NODES):
        index_data.extend([i] * rnd_array[i])

    return rnd_array, index_data


def batch_data_sub(rnd_array):
    rnd_array = rnd_array.copy()
    index_data = []
    for i in range(7):
        index_data.extend([i] * int(rnd_array[i]))

    return rnd_array, index_data


def make_path(dirname):
    if not os.path.exists("./checkpoint/" + dirname):
        os.mkdir("./checkpoint/" + dirname)
        print("Directory ", "./checkpoint/" + dirname, " Created ")
    else:
        print("Directory ", "./checkpoint/" + dirname, " already exists")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_sample', type=int)
    args = parser.parse_args()
    params['NUM_CONTAINERS_start'] = args.start_sample
    params['path'] += str(args.start_sample)
    params['rec_path'] += str(args.start_sample)

    make_path(params['path'])
    train(params)


if __name__ == "__main__":
    main()
