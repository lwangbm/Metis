"""
Environment
 (1) set the attribute of Node, App, BasicThroughput,
     Currently we fix the No. of App with 9, homogeneous cluster.
     The No. of Nodes could be set, No. of Containers in the batch is not required to know
 (2) Functions:
    1. _state_reset: clear state matrix
    2. step: allocate one container to a node, return new state
    3. get_tput_total_env: get the throughput of the entire cluster (after each episode)
"""
import numpy as np
from simulator.simulator import Simulator

class LraClusterEnv():

    def __init__(self, num_nodes, ifSimulator=True):
        #: Cluster configuration
        self.NUM_NODES = num_nodes
        # TODO: heterogeneous cluster
        self.NUM_APPS = 7
        self._state_reset()
        if ifSimulator:
            self.sim = Simulator()

    def _state_reset(self):
        self.state = np.zeros([self.NUM_NODES, self.NUM_APPS])

    def reset(self):
        self._state_reset()
        return self._get_state()

    def step(self, action, appid):
        """
        :param action: node chosen
        :param appid: current app_id of the container to be allocated
        :return: new state after allocation
        """
        curr_app = appid
        self.state[action][curr_app] += 1  # locate
        state = self._get_state()
        return state

    def _get_state(self):
        return self.state

    @property
    def _get_throughput_predictor(self):
        # testbed: using predictor
        return self.state

    def get_all_node_throughput(self):
        node_tput_list = []
        for nid in range(self.NUM_NODES):
            state_this_node = self.state[nid]
            tput_this_node = (self.sim.predict(state_this_node.reshape(1, -1)) * state_this_node).sum()
            node_tput_list.append(tput_this_node)
        return sum(node_tput_list)

    def get_tput_total_env(self):
        return self._get_throughput_predictor

    def _create_preference(self):
        from scipy.sparse import diags

        # cardinality of application itself
        # a b c d e f g h i
        # - 2 i i - 2 2 i -
        # -: cardinality == negative, i.e., anti-affinity
        # 2: cardinality == 2, if 1 < num <= cardinality, +50%, if num > cardinality, -50%
        # i: cardinatliy == infinity, i.e., affinity
        preference = diags(
            [-1, 2, np.inf, np.inf, -1, 2, 2, np.inf, -1]
        ).todense()

        # cardinality of application
        # -: abc, def, ghi
        # i: adg, beh, cfi
        # 2: af, di
        # 5: cd, fg
        # 0: others
        neg_group = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        inf_group = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        two_group = [[0, 5], [3, 8]]
        fiv_group = [[2, 3], [5, 6]]


        def assign_preference(pre, group, value):
            for g in group:
                length = len(g)
                for temp_i in range(length):
                    a = g[temp_i]
                    for temp_j in range(length):
                        if temp_i != temp_j:
                            b = g[temp_j]
                            pre[a, b] = value
                            pre[b, a] = value
            return pre

        preference = assign_preference(preference, neg_group, -1)
        preference = assign_preference(preference, inf_group, np.inf)
        preference = assign_preference(preference, two_group, 2)
        preference = assign_preference(preference, fiv_group, 5)
        preference = np.array(preference) # necessary.

        """
        Simulation: 9 nodes, 9 Apps, 81 containers
        Preference:
         [[-1. -1. -1. inf  0.  2. inf  0.  0.]
         [-1.  2. -1.  0. inf  0.  0. inf  0.]
         [-1. -1. inf  5.  0. inf  0.  0. inf]
         [inf  0.  5. inf -1. -1. inf  0.  2.]
         [ 0. inf  0. -1. -1. -1.  0. inf  0.]
         [ 2.  0. inf -1. -1.  2.  5.  0. inf]
         [inf  0.  0. inf  0.  5.  2. -1. -1.]
         [ 0. inf  0.  0. inf  0. -1. inf -1.]
         [ 0.  0. inf  2.  0. inf -1. -1. -1.]]

        # -: abc, def, ghi
        # i: adg, beh, cfi
        # 2: af, di
        # 5: cd, fg
        # 0: others
        """
        return preference

    def get_throughput_single_node(self, nid):

        state_this_node = self.state[nid]
        # TODO: we could change sum() to mean(), if using latency
        tput_this_node = (self.sim.predict(state_this_node.reshape(1, -1)) * state_this_node).sum()

        return tput_this_node, self.sim.predict(state_this_node.reshape(1, -1))

    def get_throughput_given_state(self, container_list):
        state_this_node = np.array(container_list)
        tput_this_node = (self.sim.predict(state_this_node.reshape(1, -1)) * state_this_node).sum()
        return tput_this_node, self.sim.predict(state_this_node.reshape(1, -1))