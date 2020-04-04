import numpy as np
from testbed.cluster_env import LraClusterEnv
from testbed.PolicyGradient_CPO import PolicyGradient


params = {
        # 'path': "Dynamic_large_100",
        # 'path': "Dynamic_large_100_limit10",
        # 'number of containers': 81,
        'learning rate': 0.015,
        'nodes per group': 3,
        'number of nodes in the cluster': 27,
        'container_limitation per node':8
    }

def handle_constraint(observation, NUM_NODES):

    observation_original = observation.copy()
    mapping_index = []
    # TODO: we could add more constraints here
    list_check = observation[:, :].sum(1) > params['container_limitation per node'] - 1   # >8

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

class NineNodeAPI():

    def __init__(self, path_name, surffix, path_surffix):
        """
        parameters set
        """
        self.NUM_NODES = params['number of nodes in the cluster']
        # self.NUM_CONTAINERS = params['number of containers']

        # self.sim = Simulator()
        self.env = LraClusterEnv(num_nodes=self.NUM_NODES)

        ckpt_path_1 = path_surffix + path_name + "1" + "/model.ckpt"
        ckpt_path_2 = path_surffix + path_name + "2" + "/model.ckpt"
        ckpt_path_3 = path_surffix + path_name + "3" + "/model.ckpt"
        self.nodes_per_group = int(params['nodes per group'])
        # self.number_of_node_groups = int(self.NUM_NODES / self.nodes_per_group)
        """
        Build Network
        """
        self.n_actions = self.nodes_per_group  #: 3 nodes per group
        self.n_features = int(self.n_actions * (self.env.NUM_APPS + 1 + self.env.NUM_APPS) + 1 + self.env.NUM_APPS)
        #: 29

        self.RL_1 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '1a')

        self.RL_2 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '2a')

        self.RL_3 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '3a')

        self.RL_1.restore_session(ckpt_path_1)
        self.RL_2.restore_session(ckpt_path_2)
        self.RL_3.restore_session(ckpt_path_3)

        self.observation_episode_1, self.action_episode_1, self.reward_episode_1, self.safety_episode_1 = [], [], [], []
        self.observation_optimal_1, self.action_optimal_1, self.reward_optimal_1, self.safety_optimal_1 = [], [], [], []

        self.observation_episode_2, self.action_episode_2, self.reward_episode_2, self.safety_episode_2 = [], [], [], []
        self.observation_optimal_2, self.action_optimal_2, self.reward_optimal_2, self.safety_optimal_2 = [], [], [], []

        self.observation_episode_3, self.action_episode_3, self.reward_episode_3, self.safety_episode_3 = [], [], [], []
        self.observation_optimal_3, self.action_optimal_3, self.reward_optimal_3, self.safety_optimal_3 = [], [], [], []

    def batch_data(self, rnd_array):
        index_data = []
        for i in range(7):
            index_data.extend([i] * rnd_array[i])
        return rnd_array, index_data

    def batch_data_sub(self, rnd_array):

        rnd_array = rnd_array.copy()
        index_data = []
        for i in range(7):
            index_data.extend([i] * int(rnd_array[i]))

        return rnd_array, index_data

    def store_episode_1(self, observations, actions):
        self.observation_episode_1.append(observations)
        self.action_episode_1.append(actions)

    def store_episode_2(self, observations, actions):
        self.observation_episode_2.append(observations)
        self.action_episode_2.append(actions)

    def store_episode_3(self, observations, actions):
        self.observation_episode_3.append(observations)
        self.action_episode_3.append(actions)

    def get_total_tput(self, rnd_array):

        # assert sum(rnd_array) == 81
        source_batch_, index_data = self.batch_data(rnd_array.astype(int))  # index_data = [0,1,2,0,1,2]
        env = LraClusterEnv(num_nodes=self.NUM_NODES)
        observation = env.reset().copy()  # (9,9)
        source_batch = source_batch_.copy()
        nodes_per_group = int(params['nodes per group'])
        NUM_CONTAINERS = int(sum(rnd_array))

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
            # observation_first_layer_copy = np.append(observation_first_layer_copy, ((observation_first_layer_copy[:, 2] > 0) * (observation_first_layer_copy[:, 3] > 0)).reshape(nodes_per_group, 1), axis=1)
            observation_first_layer_copy = np.array(observation_first_layer_copy).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, appid).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, np.array(source_batch_first)).reshape(1, -1)

            action_1, prob_weights = self.RL_1.choose_action_determine(observation_first_layer_copy.copy())

            observation_first_layer[action_1, appid] += 1

            # self.store_episode_1(observation_first_layer_copy, action_1)

        """
        second layer
        """
        observation_second_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20

        number_cont_second_layer = []

        for second_layer_index in range(nodes_per_group):

            rnd_array = observation_first_layer[second_layer_index].copy()
            source_batch_second, index_data = self.batch_data_sub(rnd_array)

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
                # observation_second_layer_copy = np.append(observation_second_layer_copy, ((observation_second_layer_copy[:, 2] > 0) * (observation_second_layer_copy[:, 3] > 0)).reshape(nodes_per_group, 1), axis=1)
                observation_second_layer_copy = np.array(observation_second_layer_copy).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, appid).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, np.array(source_batch_second)).reshape(1, -1)

                action_2, prob_weights = self.RL_2.choose_action_determine(observation_second_layer_copy.copy())

                observation_second_layer[action_2, appid] += 1

                # self.store_episode_2(observation_second_layer_copy, action_2)

            observation_second_layer_aggregation = np.append(observation_second_layer_aggregation, observation_second_layer, 0)

        """
        third layer
        """
        observation_third_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20
        number_cont_third_layer = []

        for third_layer_index in range(nodes_per_group * nodes_per_group):

            rnd_array = observation_second_layer_aggregation[third_layer_index].copy()
            source_batch_third, index_data = self.batch_data_sub(rnd_array)

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
                # observation_third_layer_copy = np.append(observation_third_layer_copy, ((observation_third_layer_copy[:, 2] > 0) * (observation_third_layer_copy[:, 3] > 0)).reshape(nodes_per_group, 1), axis=1)
                observation_third_layer_copy = np.array(observation_third_layer_copy).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, appid).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, np.array(source_batch_third)).reshape(1, -1)

                action_3, prob_weights = self.RL_3.choose_action_determine(observation_third_layer_copy.copy())

                observation_third_layer[action_3, appid] += 1

                # self.store_episode_3(observation_third_layer_copy, action_3)

            observation_third_layer_aggregation = np.append(observation_third_layer_aggregation, observation_third_layer, 0)

        """
        After an entire allocation, calculate total throughput, reward
        """
        env.state = observation_third_layer_aggregation.copy()
        assert sum(sum(env.state)) == NUM_CONTAINERS
        assert (env.state.sum(0) == source_batch_).all()
        """
        After an entire allocation, calculate total throughput, reward
        """
        # state = env.state
        # assert sum(sum(self.env.state)) == 81

        return env.state

