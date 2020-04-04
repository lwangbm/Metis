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
from Experiments.util.Util_Node_App import Application
from Experiments.util.Util_Node_App import Node


class LraClusterEnv():

    def __init__(self, num_nodes):
        #: Cluster configuration
        self.NUM_NODES = num_nodes # node_id: 0,1,2,...

        #: homogeneous cluster
        # TODO: heterogeneous cluster
        self.NODE_CAPACITY_NETWORK = [1200] * self.NUM_NODES
        self.NODE_CAPACITY_MEMBW = [700] * self.NUM_NODES
        self.NODE_CAPACITY_CACHE = [900] * self.NUM_NODES

        #: fixed 9 apps
        self.NUM_APPS = 7
        self.BasicThroughput = [100] * self.NUM_APPS  # normalized basic throughput

        #: Application Resource Usage Per Query
        self.NETWORK_BW_PER_QUERY = [1, 2, 3, 1, 2, 3, 1, 2, 3]  # network bandwidth
        self.MEM_BW_PER_QUERY = [1, 2, 3, 1, 2, 3, 1, 2, 3]  # memory bandwidth
        self.CACHE_PER_QUERY = [1, 2, 3, 1, 2, 3, 1, 2, 3]  # cache footprint

        #: initialized state to zero matrix
        self._state_reset()

        # testbed : initialize the predictor
        # Hyper-parameters
        # n_estimators = 20
        # max_depth = 10
        # random_state = 0
        # np_path = "./predictor/" + predictor_path + ".npz"
        # npzfile = np.load(np_path)
        # x_ori= npzfile['alloc']
        # y_ori = npzfile[predicted_metric]  # alternatively, we have npzfile['rt_50'], npzfile['rt_99']
        # y_ori = np.nan_to_num(y_ori.astype(float))
        # self.regr_ori = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
        # self.regr_ori.fit(x_ori, y_ori)  #: rps
        # self.sim = sim

    def _state_reset(self):
        self.state = np.zeros([self.NUM_NODES, self.NUM_APPS])
        # testbed : we initialize the cluster state with each node having an app_1
        # self.state[:, 1] += 1

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
    def _get_throughput(self):
        """
       First create the node instance for each node, along with the application instances residing in it
           For each node, maintain the app list, node capacity of network bw, mem bw and so on
           For each app, maintain the container number, nw/mem bw consumption for each query
       Second calculate the throughput for each app and each node, based on interference analysis
       :return: total throughput for all the nodes and all the containers residing in them
        """
        node_list = []
        for nid in range(self.NUM_NODES):
            node = Node(nid,
                        self.NODE_CAPACITY_NETWORK[nid],
                        self.NODE_CAPACITY_MEMBW[nid],
                        self.NODE_CAPACITY_CACHE[nid],
                        self)
            for aid in range(self.NUM_APPS):
                num_container = self.state[nid][aid]
                if num_container > 0:
                    app = Application(aid,
                                      self.BasicThroughput[aid],
                                      self.NETWORK_BW_PER_QUERY[aid],
                                      self.MEM_BW_PER_QUERY[aid],
                                      self.CACHE_PER_QUERY[aid],
                                      num_container)
                    node.add_application(app)
            node.calculate_new_tput()
            node_list.append(node)

        total_tput = 0
        for node in node_list:
            total_tput += node.total_tput()
        return total_tput

    @property
    def _get_throughput_predictor(self):
        # testbed: using predictor

        return self.state

        # return (self.sim.predict(self.state.reshape(self.NUM_NODES, -1)) * self.state).sum()
        # node_tput_list = []
        #
        # for nid in range(self.NUM_NODES):
        #     state_this_node = self.state[nid]
        #     # TODO: we could change sum() to mean(), if using latency
        #     tput_this_node = (self.sim.predict(state_this_node.reshape(1, -1)) * state_this_node).sum()
        #     node_tput_list.append(tput_this_node)
        # return sum(node_tput_list)




    def get_tput_total_env(self):
        # testbed: we replace env.get_tput_total_env() with the predictor
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


    def get_min_throughput(self):
        # TODO: minimum throughput, not used yet, for fairness
        node_list = []
        for nid in range(self.NUM_NODES):
            node = Node(nid,
                        self.NODE_CAPACITY_NETWORK[nid],
                        self.NODE_CAPACITY_MEMBW[nid],
                        self.NODE_CAPACITY_CACHE[nid],
                        self)
            for aid in range(self.NUM_APPS):
                container = self.state[nid][aid]
                if container > 0:
                    app =  Application(aid,
                                      100,
                                      self.NETWORK_BW_PER_QUERY[aid],
                                      self.MEM_BW_PER_QUERY[aid],
                                      self.CACHE_PER_QUERY[aid],
                                      container)
                    node.add_application(app)
            node.calculate_new_tput()
            node_list.append(node)

        minimum_tput = 100
        for node in node_list:
            if node.minimum() < minimum_tput:
                minimum_tput = node.minimum()
        return minimum_tput

    def get_throughput_single_node(self, nid):
        # testbed: we replace env.get_tput_total_env() with the predictor

        state_this_node = self.state[nid]
        # TODO: we could change sum() to mean(), if using latency
        tput_this_node = (self.sim.predict(state_this_node.reshape(1, -1)) * state_this_node).sum()

        return tput_this_node, self.sim.predict(state_this_node.reshape(1, -1))

    def get_throughput_given_state(self, container_list):
        state_this_node = np.array(container_list)
        tput_this_node = (self.sim.predict(state_this_node.reshape(1, -1)) * state_this_node).sum()
        return tput_this_node, self.sim.predict(state_this_node.reshape(1, -1))

    def get_SLA_violation(self, sla):
        violation = 0
        for nid in range(self.NUM_NODES):
            state_this_node = self.state[nid]
            # TODO: we could change sum() to mean(), if using latency
            tput_this_node = (self.sim.predict(state_this_node.reshape(1, -1)))
            for app_index in range(self.NUM_APPS):
                if tput_this_node[0][app_index] >0:
                    if tput_this_node[0][app_index] < sla:
                        violation += state_this_node[app_index]
        return violation