import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
from Experiments.cluster_env import LraClusterEnv
from pathlib import Path
import csv

def argmin_random(array, filter=None, random=1):
    assert type(array)==np.ndarray, "Only support np.ndarray"
    if filter is not None:
        if filter.all() == False:
            for i in range(len(filter)):
                if filter[i] == False:
                    # print("ARGMIN: ", array, filter)
                    array[i] = np.inf  # makes it impossible to be min()
    if not random:
        node_id = array.argmin()
    else:
        arg_array = np.where(array == array.min())[0]
        node_id = np.random.choice(arg_array)
    return node_id

def argmax_random(array, filter=None, random=1):
    assert type(array)==np.ndarray, "Only support np.ndarray"
    if filter is not None:
        if filter.all() == False:
            for i in range(len(filter)):
                if filter[i] == False:
                    # print("ARGMAX: ", array, filter)
                    array[i] = -np.inf  # makes it impossible to be max()
    if not random:
        node_id = array.argmax()
    else:
        arg_array = np.where(array == array.max())[0]
        node_id = np.random.choice(arg_array)
    return node_id

class LraClusterEnvSim(LraClusterEnv):
    def predict(self, in_array):
        """
        sim: sim.predict(np.array(num_cntr_app)) => np.array(QoS)
        num_cntr_app: shape: (?, 9), int
        QoS: shape: (?, 9), float
        e.g., microbench.simulator.simulator.Simulator
        """
        out_array = []
        for single_alloc in in_array:
            if single_alloc.shape != (9,):
                print(single_alloc)
                exit()
            result = self._predict(single_alloc)
            out_array.append(result)
        if len(out_array) != len(in_array):
            print(in_array)
            print(out_array)
            exit()
        return np.array(out_array)

    def _predict(self, alloc):
        tput_this_node, tput_breakdown = self.get_throughput_given_state(alloc.reshape(-1, len(alloc)))
        print(tput_breakdown)
        return tput_breakdown[0]


class ParagonScheduler():

    def __init__(self, num_nodes, rnd_array, sim=None, sensitivity=0.95, node_capacity=10, num_sois=4, verbose=0):
        self.num_nodes = num_nodes
        self.rnd_array = rnd_array # np.array, shape: (9, )
        # e.g., array([7, 7, 7, 7, 7, 7, 7, 7, 7]) for num_container==63 case. (referring `batch_data()` method in performance_test.py)
        self.sim = sim # simulator that returns soi info
        self.num_sois = num_sois # (Simulator) source of interferences, i.e., resource that matters
        self.node_capacity=node_capacity # (Simulator)
        self.qos_threshold=sensitivity # (Simulator, e.g. 0.95)
        self.sensitivity = sensitivity # (by sim)
        self.verbose=verbose

        self.num_apps = self.rnd_array.shape[0]
        self.num_container = self.rnd_array.sum()
        self.index_data = [] # list of int, sorted, len: self.num_container
        for app_index in range(self.num_apps):
            self.index_data.extend([app_index] * rnd_array[app_index])
        # e.g., 63 cntr cases: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8] (referring `batch_data()` method in performance_test.py)

        self.tolerate = None
        self.cause = None
        if self.verbose: print("Initializing Paragon Scheduler, calculating interference scores ...")
        self.calc_interference_scores()
        if self.verbose: print("\nFINAL:\ntolerate:\n{}\ncause\n{}\n=============== __init__ DONE ===============\n".format(self.tolerate, self.cause))

        self.state = np.zeros((self.num_nodes, self.num_apps))

    def choose_action(self, app_id, state):
        return_node_id = 0
        if self.verbose: print("Candidate: {}".format(app_id))
        has_capacity = np.sum(state, axis=1) < self.node_capacity  # to select from, (num_nodes, ), still available
        # it should be "np.sum(state, axis=1) < self.node_capacity"  if observation is the current observation
        # it is changed to "<=" since the observation in clean_sla_testbed is after "observation[:, index_data[inter_episode_index]] += 1", i.e., the state in imagination.
        if has_capacity.any() == False:
            print("No capacity!")
            print(np.sum(state, axis=1))
            print(self.node_capacity)
            exit()
        # prepare
        allnode_tolerate_allsoi = self.get_all_node_tolerate(state)  # shape: (num_nodes, num_sois)
        allnode_cause_allsoi = self.get_all_node_cause(state)  # shape: (num_nodes, num_sois)
        if self.verbose > 1: print("DEBUG allnode_tolerate_allsoi:\n{}\nDEBUG allnode_cause_allsoi:\n{}".format(allnode_tolerate_allsoi, allnode_cause_allsoi))
        app_cause_allsoi = self.get_cause()[app_id]  # shape: (num_SoIs, )
        app_cause_argsort = np.flip(app_cause_allsoi.argsort())  # largest -> smallest
        app_tolerate_allsoi = self.get_tolerate()[app_id]  # shape: (num_SoIs, )
        app_tolerate_argsort = app_tolerate_allsoi.argsort()  # smallest -> largest

        # to schedule, D1
        node_candidate = np.ones(self.num_nodes, dtype=int)  # to select from, (num_nodes, )

        for soi_c, soi_t in zip(app_cause_argsort, app_tolerate_argsort):
            # D1 = t_server − c_newapp
            # "We query the server state and select the server set for which D1 is non-negative for this SoI"
            app_cause_soi = app_cause_allsoi[soi_c]  # c_newapp, scalar
            node_tolerate_soi = allnode_tolerate_allsoi[:, soi_c]  # t_server, (num_nodes, )
            D1_allnode = node_tolerate_soi - app_cause_soi  # D1 = t_server − c_newapp, (num_nodes, )
            node_candidate_update = (D1_allnode >= 0) * node_candidate  # filter out negative D1
            if self.verbose > 1: print("DEBUG CAUSE: D1_allnode: {}".format(D1_allnode * node_candidate))
            if ~(node_candidate_update.any()):  # all zeros
                node_id = argmax_random(D1_allnode, filter=has_capacity)  # node_id: the one with largest D1 (negative, closest to zero)
                if self.verbose: print("CAUSE check: SoI #{}: {}\n          -> {}, returns #{} to allocate\n".format(soi_c, node_candidate, D1_allnode, node_id))
                return_node_id = node_id
                node_candidate_update = np.zeros(self.num_nodes, dtype=int) # return flag
                break
            else:
                node_candidate = node_candidate_update
                if self.verbose: print(
                    "CAUSE check: SoI #{}: {} => {}".format(soi_c, node_candidate, node_candidate_update))

            # D2 = t_newapp − c_server
            if ~(node_candidate_update.any()):
                # "[ERROR] Empty subset before entering D2: soi_c {}, soi_t {}."
                state_sum_alloc = np.sum(state, axis=1)
                node_id = argmin_random(state_sum_alloc, filter=has_capacity)
                break
            app_tolerate_soi = app_tolerate_allsoi[soi_t]  # t_newapp, scalar
            node_cause_soi = allnode_cause_allsoi[:, soi_t]  # c_server, (num_nodes, )
            D2_allnode = - node_cause_soi + app_tolerate_soi  # D2 = t_newapp − c_server, (num_nodes, )
            node_candidate_update = (D2_allnode >= 0) * node_candidate  # filter out negative D2
            if self.verbose > 1: print("DEBUG CAUSE: D2_allnode: {}".format(D2_allnode * node_candidate))
            if ~(node_candidate_update.any()):  # all zeros
                node_id = argmax_random(D2_allnode, filter=has_capacity)  # node_id: the one with largest D2 (negative, closest to zero)
                if self.verbose: print(
                    "TOLER check: SoI #{}: {}\n          -> {}, returns #{} to allocate\n".format(soi_t, node_candidate, D2_allnode,  node_id))
                return_node_id = node_id
                break
            else:
                node_candidate = node_candidate_update
                if self.verbose: print(
                    "TOLER check: SoI #{}: {} => {}".format(soi_t, node_candidate, node_candidate_update))

        if ~(node_candidate_update.any()) == True:
            state_sum_alloc = np.sum(state, axis=1)
            node_id = argmin_random(state_sum_alloc, filter=has_capacity)
            # select_server_with_best_sc(): raise NotImplementedError
        else:
            D1_allnode[D1_allnode <= 0] = np.inf
            D2_allnode[D2_allnode <= 0] = np.inf
            D1_D2_l1norm = np.clip(D1_allnode + D2_allnode, 0, None)  # ignore the negative ones
            node_id = argmin_random(D1_D2_l1norm, filter=has_capacity)
            if self.verbose: print(
                "FINAL check: {} + {}\n          => {}, returns #{} to allocate\n".format(D1_allnode, D2_allnode, D1_D2_l1norm, node_id))

        return_node_id = node_id
        # state[return_node_id , app_id] += 1
        # assert ((np.sum(state, axis=1) <= self.node_capacity).all())

        # except AssertionError:
        #     state_sum_alloc = np.sum(state, axis=1)
        #     if self.verbose: print('state sum:', state_sum_alloc)
        #     node_id = argmin_random(state_sum_alloc)
        #     if self.verbose: print("\n[ERROR] Capacity violate, load-balancing allocation: app {} -> node {}".format(app_id, node_id))
        #     return_node_id = node_id

        # except Exception as e:
        #     if self.verbose: print(e)
        #     state_sum_alloc = np.sum(state, axis=1)
        #     # if self.verbose: print('state sum:', state_sum_alloc)
        #     # node_id = np.random.randint(self.num_nodes)  # random
        #     # while has_capacity[node_id] == False:
        #     #     node_id = np.random.randint(self.num_nodes)
        #     node_id = argmin_random(state_sum_alloc)
        #     if self.verbose: print("{}\n[ERROR] fallback to load-balancing allocation: app {} -> node {}".format(e, app_id, node_id))
        #     return_node_id = node_id

        self.state[return_node_id, app_id] += 1
        return return_node_id


    def sensitivity_rescale(self, temp_tolerate, temp_cause, sensitivity=None):
        if sensitivity is None:
            sensitivity = self.sensitivity # e.g., 10
        if sensitivity < 1:
            print("Invalid sensitivity, should >= 1")
            return temp_tolerate, temp_cause

        tolerate = ((temp_tolerate + sensitivity - 1 ) / sensitivity)
        cause = temp_cause / sensitivity
        return tolerate, cause

    def calc_interference_scores(self, sim=None, app_list=None, node_capacity=None, qos_threshold=None):
        if sim is None:
            sim = self.sim
        if app_list is None:
            app_list = list(range(self.num_apps))
        if node_capacity is None:
            node_capacity = self.node_capacity
        if qos_threshold is None:
            qos_threshold = self.qos_threshold

        # for different types of 'sim'
        if type(sim) == LraClusterEnv:
            if self.verbose: print("Using LraClusterEnv for resource contentions ...")
            # self.tolerate, self.cause = self._env_interference_scores()
            sim=LraClusterEnvSim(1)
            tolerate, cause = self._app_interference_scores(sim, app_list, node_capacity, qos_threshold, verbose=0)
        elif type(sim) == None:
            if self.verbose: print("Using Simulator for app interference scores ...")
            tolerate, cause = self._app_interference_scores(sim, app_list, node_capacity, qos_threshold, verbose=0)
            # 0: Redis-1, 4: Sysbench-MEM, 5: Sysbench-FILEIO, 8: Sysbench-CPU
            # SoI: Network; Memory; Disk; CPU
            self.tolerate, self.cause = tolerate[:,(0,4,5,8)], cause[:,(0,4,5,8)]
        elif type(sim) == str: # i.e., csvpath
            csvfilepath = Path('.') / sim
            if self.verbose: print('loading csv from ', csvfilepath.resolve())
            num_apps = self.num_apps
            self.num_sois = self.num_apps
            temp_tolerate = np.ones((num_apps, num_apps))
            temp_cause = np.zeros((num_apps, num_apps))

            counter = 0
            with open(csvfilepath, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                header = next(reader)
                for row in reader:
                    app_s, app_c, less_or_lager, threshold = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    if less_or_lager != 0:
                        continue

                    temp_tolerate[app_s-1, app_c-1] = threshold / self.node_capacity
                    temp_cause[app_c-1, app_s-1] = max(1 - temp_tolerate[app_s-1, app_c-1], 0)
                    counter += 1

            self.tolerate, self.cause = self.sensitivity_rescale(temp_tolerate, temp_cause, self.sensitivity)
            # self.tolerate = temp_tolerate.copy()
            # self.cause = temp_cause.copy()
        else:
            print("[WRONG] unacceptable simulator type:{}".format(type(sim)))
            exit()

        if (-1 in self.tolerate) or (-1 in self.cause):
            self._reconstruct_interference_scores()


    def _env_interference_scores(self):
        """
        Idea: using XXX_PER_QUERY as reference for resource contention
        """
        env = self.sim
        cap_netbw = env.NODE_CAPACITY_NETWORK[0]
        cap_membw = env.NODE_CAPACITY_MEMBW[0]
        cap_cache = env.NODE_CAPACITY_CACHE[0]
        basic_tput = env.BasicThroughput[0]
        netbw = np.array(env.NETWORK_BW_PER_QUERY)
        membw = np.array(env.MEM_BW_PER_QUERY)
        cache = np.array(env.CACHE_PER_QUERY)
        assert cache.shape==(9,), "number of app resource usage per query not correct"
        netbw_cause = np.array(env.NETWORK_BW_PER_QUERY) * basic_tput / cap_netbw
        membw_cause = np.array(env.MEM_BW_PER_QUERY) * basic_tput / cap_membw
        cache_cause = np.array(env.CACHE_PER_QUERY) * basic_tput / cap_cache

        cause = np.vstack((netbw_cause,membw_cause,cache_cause)).T  # shape: (3, 9)
        tolerate = 1-cause
        return tolerate, cause

    def _app_interference_scores(self, sim, app_list=[0], node_capacity=8, qos_threshold=0.95, verbose=0):
        """ params
        sim: sim.predict(np.array(num_cntr_app)) => np.array(QoS)
            num_cntr_app: shape: (?, 9), int
            QoS: shape: (?, 9), float
            e.g., microbench.simulator.simulator.Simulator

        app_list: list of app index (0 ~ self.num_apps-1) to profile, len: 1 ~ self.num_apps
            e.g., [0, 1, 3, 5, 7, 8] means only these 6 apps have been profiled
            e.g., [0,1,2,3,4,5,6,7,8] means all the groundtruth is collected

        node_capacity: int, the maximum number of containers that can be well-isolated in the server.
            e.g., 8, for m5.4xlarge with 16vCPU cores; each container occupies 2vCPU cores and 8GB RAM.

        qos_threshold: 0 ~ 1, float, the least acceptable QoS marked as not affected by the interference.
            e.g., 0.95, means if the rps < 95% baseline rps, it will be marked as tolerating/causing interference.

        """

        """
        The physical meaning of 'tolerate' and 'cause' has neither been clearly stated in Paragon, nor in Quasar. The authors mentioned "sensitivity score over 60%", but I do not see how this '60%' be derived. All the sentences, as I could found useful in 2.3 section, are cited as follows.

        ref. 2.3 Classification for Interference
        "To derive sensitivity scores we develop several microbenchmarks, each stressing a specific shared resource with tunable intensity. We run an application concurrently with a microbenchmark and progressively tune up its intensity until the application violates its QoS, which is set at 95% of the performance achieved in isolation. Applications with high tolerance to interference (e.g., sensitivity score over 60% are easier to co-schedule than applications with low tolerance (low sensitivity score). Similarly, we detect the sensitivity of a microbenchmark to the interference the application causes by tuning up its intensity and recording when the performance of the microbenchmark degrades by 5% compared to its performance in isolation. In this case, high sensitivity scores, e.g., over 60% correspond to applications that cause a lot of interference in the specific shared resource."

        """
        # init as -1, means unprofiled.
        tolerate=-np.ones([self.num_apps, self.num_apps])
        cause=-np.ones([self.num_apps, self.num_apps])

        for tgt in app_list:  # target application
            for bgd in app_list:  # background application, i.e., microbenchmark
                # ignore cases that already been profiled
                if (tolerate[tgt, bgd]!=-1) and (tolerate[bgd, tgt]!=-1) and (cause[tgt, bgd]!=-1) and (cause[bgd, tgt]!=-1):
                    continue

                # build baseline
                baseline_input = np.zeros(9, dtype=int)
                baseline_input[tgt] = 1  # [0,0,1,0,0,0,0,0,0]
                baseline_output = sim.predict(baseline_input.reshape(-1, 9))[0]
                tgt_base = baseline_output[tgt]
                if verbose: print("\nApp #{} baseline rps: {:.4f}\nWith bgd #{} increasing ...".format(tgt, tgt_base, bgd))

                # increase bgd ( tolerate(tgt,bgd), cause(bgd,tgt) )
                temp_tolerate = None
                for bgd_cntr in range(1, node_capacity):  # 1 ~ node-capacity-1 "progressively tune up its intensity"
                    test_input = baseline_input.copy()
                    test_input[bgd] = bgd_cntr  # [0,0,1,0,bgd_cntr,0,0,0,0]
                    test_output = sim.predict(test_input.reshape(-1, 9))[0]
                    tgt_test = test_output[tgt]
                    if verbose: print("With {} bgd cntrs, tgt rps: {:.4f} ({:.2f}% baseline)".format(bgd_cntr, tgt_test, 100 * tgt_test / tgt_base))
                    if tgt_test / tgt_base < qos_threshold:  # "until the application violates its QoS"
                        temp_tolerate = (bgd_cntr - 1) / (node_capacity - 1)  # roll back to bgd_cntr - 1 situation. If anti-affinity, probably become (1 - 1)/(8 - 1) = 0, i.e., no tolerate of bgd.
                        break
                if temp_tolerate is None: # in case no violation happens in the loop
                    temp_tolerate = 1
                    bgd_cntr = node_capacity
                temp_cause = 1 - temp_tolerate
                tolerate[tgt, bgd] = temp_tolerate
                cause[bgd, tgt] = temp_cause
                if verbose: print("tolerate[{}, {}] = {:.4f} ({}/{})\n   cause[{}, {}] = {:.4f} (1-{}/{})".format(tgt, bgd, temp_tolerate, bgd_cntr-1, node_capacity-1, bgd, tgt, temp_cause, bgd_cntr-1, node_capacity-1))

        # if verbose:
            # print("\nFINAL:\ntolerate")
            # print(tolerate)
            # print("\ncause")
            # print(cause)

        return tolerate, cause
        # both tolerate and cause: nd.array, shape: (num_apps, num_apps)

        """ Settings and Examples.
        (the "workload" is ambiguous, let's make it resource. But actually in our microbenchmark, if one deploys 4 more containers, though 4 more resource will be allocated, the workload will also become 4 times; it aligns with Paragon's definition as well.
        Having another struggle is how to set the baseline, should that be target and microbenchmark with each 2 vCPU allocated? or target has 2, microbenchmark has 0? or target has 8 and microbenchmark has 8? Finally I choose [target, microbenchmark] = [2, 0] as baseline, representing perfect isolation)

        Say, we have microbenchmark and target app co-locate on a machine with 16 vCPU cores. The baseline is microbenchmark taking 0 vCPU and target app taking 2 vCPU. The idle resource under good isolation is 14 vCPU cores.

        If microbenchmark takes 8 vCPU, (57.1% of idle resource), the target app still takes 2 vCPU as baseline (if here it takes 8 vCPU, which fully occupies the cluster, we are losing the baseline), but target app's QoS (e.g., RPS) degrades to 95%, we mark the tolerate[target app, microbenchmark] = 57.1%.

        On the other way around, if microbenchmark takes 2 vCPU, and the target takes 8 vCPU (57.1% of all idle vCPU cores in sharing),  which makes the microbenchmark violate its 95% QoS, we mark the cause[target app, microbenchmark] = 1 - 57.1% = 42.9%.

        In the extreme cases, if the isolation is perfect, even the microbenchmark takes 100% of the idle resources, i.e., 14 vCPU cores, the target app still gets >95% QoS, tolerate[target app, microbenchmark] = 100%; vice versa, if target app taking 100% of the idle resources still cannot downgrade microbenchmark's QoS to 95%, cause[target app, microbenchmark] = 1 - 100% = 0%.
        """

    def get_cause(self):
        if self.cause is None:
            self.calc_interference_scores()
        return self.cause.copy()  # read-only, shape: (num_apps, num_apps)

    def get_tolerate(self):
        if self.tolerate is None:
            self.calc_interference_scores()
        return self.tolerate.copy()  # read-only, shape: (num_apps, num_apps)

    def get_node_tolerate(self, node_id, state):
        node_app_alloc = state[node_id].copy()  # shape: (num_apps, ) e.g.,
        app_retrieve_index = np.where(node_app_alloc >= 1)  # shape: (num_non_zero_in_node_app_alloc, )
        app_soi_tolerate = self.get_tolerate()[app_retrieve_index]  # shape: (num_non_zero_in_node_app_alloc, num_sois)
        if len(app_soi_tolerate) == 0: # fix bug in all zero cases
            return np.ones(self.num_sois) # full tolerance
        else:
            node_soi_tolerate = app_soi_tolerate.min(axis=0) # shape: (num_sois, )
            # the minimum of sensitivities of individual applications on it
            return node_soi_tolerate

    def get_node_cause(self, node_id, state):
        node_app_alloc = state[node_id].copy()  # shape: (num_apps, ) e.g.,
        """WAS: only differentiate app exist/non-exist
        app_retrieve_index = np.where(node_app_alloc >= 1)  # shape: (num_non_zero_in_node_app_alloc, )
        app_soi_cause = self.get_cause()[app_retrieve_index]  # shape: (num_non_zero_in_node_app_alloc, num_sois)
        node_soi_cause = app_soi_cause.sum(axis=0) # shape: (num_sois, )
        """
        node_soi_cause = np.matmul(node_app_alloc, self.get_cause())  # (num_apps, ) * (num_apps, num_sois) = (num_sois, )
        # the sum of sensitivities of individual applications on it
        return node_soi_cause

    def get_all_node_tolerate(self, state):
        all_node_soi_tolerate = []
        for node_id in range(self.num_nodes):
            all_node_soi_tolerate.append(self.get_node_tolerate(node_id, state))
        return np.array(all_node_soi_tolerate)  # shape: (num_nodes, num_sois)

    def get_all_node_cause(self, state):
        all_node_soi_cause = []
        for node_id in range(self.num_nodes):
            all_node_soi_cause.append(self.get_node_cause(node_id, state))
        return np.array(all_node_soi_cause)  # shape: (num_nodes, num_sois)

    def _reconstruct_interference_scores(self):
        raise NotImplementedError("Not fully profiled is not considered for now")

    def calc_heterogeneity_scores(self):
        raise NotImplementedError("Scheduling in heterogeneous clusters is not considered for now")
