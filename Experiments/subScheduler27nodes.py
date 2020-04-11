import numpy as np
from cluster_env import LraClusterEnv
from PolicyGradient import PolicyGradient


params = {
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


class subScheduler():

    def __init__(self, path_name, surffix, path_surffix):
        """
        parameters set
        """
        self.NUM_NODES = params['number of nodes in the cluster']
        self.NUM_APPS = 7
        # self.NUM_CONTAINERS = params['number of containers']

        # self.sim = Simulator()
        # self.env = LraClusterEnv(num_nodes=self.NUM_NODES)

        ckpt_path_1 = path_surffix + path_name + "_1" + "/model.ckpt"
        ckpt_path_2 = path_surffix + path_name + "_2" + "/model.ckpt"
        ckpt_path_3 = path_surffix + path_name + "_3" + "/model.ckpt"
        self.nodes_per_group = int(params['nodes per group'])
        # self.number_of_node_groups = int(self.NUM_NODES / self.nodes_per_group)
        """
        Build Network
        """
        self.n_actions = self.nodes_per_group  #: 3 nodes per group
        self.n_features = int(self.n_actions * self.NUM_APPS + 1 + self.NUM_APPS)  #: 29

        self.RL_1 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '1')

        self.RL_2 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '2')

        self.RL_3 = PolicyGradient(n_actions=self.n_actions, n_features=self.n_features, learning_rate=params['learning rate'], suffix=surffix + '3')

        self.RL_1.restore_session(ckpt_path_1)
        self.RL_2.restore_session(ckpt_path_2)
        self.RL_3.restore_session(ckpt_path_3)

    def batch_data(self, rnd_array):
        index_data = []
        for i in range(7):
            index_data.extend([i] * rnd_array[i])
        return rnd_array, index_data

    def get_total_tput(self, rnd_array):

        # assert sum(rnd_array) == 81
        source_batch, index_data = self.batch_data(rnd_array.astype(int))  # index_data = [0,1,2,0,1,2]
        env = LraClusterEnv(num_nodes=self.NUM_NODES, ifSimulator=False)
        observation = env.reset().copy()  # (9,9)


        """
        Episode
        """
        for inter_episode_index in range(int(sum(rnd_array))):
            # observation_new_list = []
            # observation[:, index_data[inter_episode_index]] += 1
            source_batch[index_data[inter_episode_index]] -= 1
            observation, mapping_index = handle_constraint(observation, self.NUM_NODES)

            assert len(mapping_index) > 0

            observation_first_layer = np.empty([0, env.NUM_APPS], int)
            number_of_first_layer_nodes = int(self.NUM_NODES / self.nodes_per_group)  # 9
            for i in range(self.nodes_per_group):
                observation_new = np.sum(observation[i * number_of_first_layer_nodes:(i + 1) * number_of_first_layer_nodes], 0).reshape(1, -1)
                observation_first_layer = np.append(observation_first_layer, observation_new, 0)
            observation_first_layer[:, index_data[inter_episode_index]] += 1
            observation_first_layer = np.append(np.append(observation_first_layer, index_data[inter_episode_index]), np.array(source_batch)).reshape(1,-1)
            # observation_first_layer = np.array(observation_first_layer).reshape(1, -1)
            # observation_first_layer = np.append(observation_first_layer, index_data[inter_episode_index]).reshape(1, -1)
            # observation_first_layer = np.append(observation_first_layer, np.array(source_batch)).reshape(1, -1)  # (1,29)

            action_1, prob_weights = self.RL_1.choose_action_determine(observation_first_layer)

            observation_copy = observation
            observation_copy = observation_copy[action_1 * number_of_first_layer_nodes: (action_1 + 1) * number_of_first_layer_nodes]
            number_of_second_layer_nodes = int(number_of_first_layer_nodes / self.nodes_per_group)  # 9/3 = 3
            observation_second_layer = np.empty([0, env.NUM_APPS], int)
            for i in range(self.nodes_per_group):
                observation_new = np.sum(observation_copy[i * number_of_second_layer_nodes:(i + 1) * number_of_second_layer_nodes], 0).reshape(1, -1)
                observation_second_layer = np.append(observation_second_layer, observation_new, 0)
            observation_second_layer[:, index_data[inter_episode_index]] += 1
            observation_second_layer = np.append(np.append(observation_second_layer, index_data[inter_episode_index]), np.array(source_batch)).reshape(1,-1)

            # observation_second_layer = np.array(observation_second_layer).reshape(1, -1)
            # observation_second_layer = np.append(observation_second_layer, index_data[inter_episode_index]).reshape(1, -1)
            # observation_second_layer = np.append(observation_second_layer, np.array(source_batch)).reshape(1, -1)
            action_2, prob_weights = self.RL_2.choose_action_determine(observation_second_layer)

            observation_copy = observation_copy[action_2 * number_of_second_layer_nodes: (action_2 + 1) * number_of_second_layer_nodes]
            number_of_third_layer_nodes = int(number_of_second_layer_nodes / self.nodes_per_group)  # 3/3 = 1
            observation_third_layer = np.empty([0, env.NUM_APPS], int)
            for i in range(self.nodes_per_group):
                observation_new = np.sum(observation_copy[i * number_of_third_layer_nodes:(i + 1) * number_of_third_layer_nodes], 0).reshape(1, -1)
                observation_third_layer = np.append(observation_third_layer, observation_new, 0)
            observation_third_layer[:, index_data[inter_episode_index]] += 1
            observation_third_layer = np.append(np.append(observation_third_layer, index_data[inter_episode_index]), np.array(source_batch)).reshape(1,-1)

            # observation_third_layer = np.array(observation_third_layer).reshape(1, -1)
            # observation_third_layer = np.append(observation_third_layer, index_data[inter_episode_index]).reshape(1, -1)
            # observation_third_layer = np.append(observation_third_layer, np.array(source_batch)).reshape(1, -1)

            action_3, prob_weights = self.RL_3.choose_action_determine(observation_third_layer)

            final_decision = action_1 * number_of_first_layer_nodes + action_2 * number_of_second_layer_nodes + action_3 * number_of_third_layer_nodes

            appid = index_data[inter_episode_index]
            # observation_ = env.step(action*nodes_per_group + Node_index[action], appid)
            observation_ = env.step(mapping_index[final_decision], appid)
            observation = observation_.copy()  # (9,9)
        """
        After an entire allocation, calculate total throughput, reward
        """
        state = env.get_tput_total_env()
        # assert sum(sum(self.env.state)) == 81

        return state

