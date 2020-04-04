"""
class Application: maintain several throughput limitations,
    e.g., baseThroughput, affinityThroughput, anti-affinityThroughput (bwThroughput, cacheThroughput, memBwThroughput)


class Node: Node state,
    (1) maintain the Application list;
    (2) calculate throughput on this node, based on resource contention and matrix-based affinity (calculate_new_tput())
"""
class Application:
    def __init__(self, applicationName, baseThroughput,
                 networkBwPerQuery, memBwPerQuery, cachePerQuery, numberofContainers):
        """

        :param applicationName:
        :param baseThroughput:
        :param networkBwPerQuery:
        :param memBwPerQuery:
        :param cachePerQuery:
        :param numberofContainers:
        """
        self.applicationName = applicationName  # app id
        # self.applicationType = mytype  # app type, not used now
        self.baseThroughput = baseThroughput  # base throughput value, without interference
        self.networkBwPerQuery = networkBwPerQuery  # network bw consuming per query, showing the network resource contention
        self.memBwPerQuery = memBwPerQuery  # mem bw consuming per query, showing the network resource contention
        self.cachePerQuery = cachePerQuery
        self.numberofContainers = numberofContainers  # number of containers of this app located on the current node

        self.throughput = 0  # eventual tput after calculation
        self.affinityThroughput = 0  # temporary tput value, based on affinity analysis
        self.bwThroughput = 0  # temporary tput value, based on network contention analysis
        self.memBwThroughput = 0  # temporary tput value, based on mem contention analysis
        self.cacheThroughput = 0 # temporary tput value, based on cache contention analysis


class Node:
    def __init__(self, name, nwbwlimit, membwlimit, cacheLimit, env):
        """
        :param name: node id
        :param nwbwlimit: network bw capacity of this node
        :param membwlimit: mem bw capacity of this node
        :param env: env variable, used to get the env.state matrix, and preference.
        """
        self.name = name
        self.networkBwLimitation = nwbwlimit
        self.memBwLimit = membwlimit
        self.cacheLimit = cacheLimit
        self.applicationList = []
        self.env = env

    def add_application(self, newapp):
        self.applicationList.append(newapp)

    def calculate_new_tput(self):
        """
        Main function:
        calculate throughput on this node, based on resource contention and matrix-based affinity
        :return:
        """
        self._check_inter_affinity()
        self._check_net_bw_limit()
        self._check_mem_bw_limit()
        self._check_cache_limit()
        self._update_tput()

    def _check_inter_affinity(self):
        """ check affinity using preference matrix """
        preference = self.env._create_preference()


        nid = self.name

        # intra- and inter- constraints
        for app in self.applicationList:
            aid = app.applicationName
            # revise 1 app's throughput at a time
            aid_constraints = preference[aid][:]

            nid_state = self.env.state[nid]
            # inter-cardinality
            exceed_card = (nid_state[:] > aid_constraints) & (aid_constraints != 0) & (nid_state[:] > 0)  # punish
            within_card = (nid_state[:] <= aid_constraints) & (nid_state[:] > 0)  # reward

            # intra-cardinality
            self_card = aid_constraints[aid]
            ## if run <=1 containers, or indifference
            if nid_state[aid] <= 1 or self_card == 0:
                within_card[aid] = False  # no reward
                exceed_card[aid] = False  # no punish
            ## anti-affinity
            elif self_card < 0 or self_card == 1:
                # represent self-anti-affinity by <0 value or ==1
                exceed_card[aid] = True
                within_card[aid] = False
            ## cardinality
            else:
                exceed_card[aid] = nid_state[aid] > self_card
                within_card[aid] = nid_state[aid] <= self_card

            tput_change = (-sum(exceed_card) * 0.5) + (sum(within_card) * 0.5)
            """
            Add additional affinity/anti-affinity
            """
            # if nid_state[7] * nid_state[8] >0:
            #      tput_change = 2
            # if nid_state[0] * nid_state[1] >0 and (aid == 1 or aid == 0):
            #     tput_change = 10
            app.affinityThroughput = max(app.baseThroughput * (1 + tput_change),0)

    def _check_net_bw_limit(self):
        """ resource contention check, network resource """
        consumed_bw = 0
        for app in self.applicationList:
            consumed_bw += app.affinityThroughput * app.numberofContainers * app.networkBwPerQuery

        if consumed_bw > self.networkBwLimitation:

            number_container = 0
            for app in self.applicationList:
                number_container += app.numberofContainers
            bw_per_conatainer = self.networkBwLimitation / number_container
            for app in self.applicationList:
                app.bwThroughput = bw_per_conatainer / app.networkBwPerQuery
        else:
            for app in self.applicationList:
                app.bwThroughput = app.affinityThroughput

    def _check_mem_bw_limit(self):
        """ resource contention check, memory bandwidth resource """
        consumed_mem_bw = 0
        for app in self.applicationList:
            consumed_mem_bw += app.affinityThroughput * app.numberofContainers * app.memBwPerQuery

        if consumed_mem_bw > self.memBwLimit:
            number_container = 0
            for app in self.applicationList:
                number_container += app.numberofContainers
            # fair memory bandwidth allocation among containers, not applications
            mem_bw_per_container = self.memBwLimit / number_container
            for app in self.applicationList:
                app.memBwThroughput = mem_bw_per_container / app.memBwPerQuery
        else:
            for app in self.applicationList:
                app.memBwThroughput = app.affinityThroughput

    def _check_cache_limit(self):
        """ resource contention check, cache space resource """
        consumed_cache = 0
        for app in self.applicationList:
            consumed_cache += app.affinityThroughput * app.numberofContainers * app.cachePerQuery

        if consumed_cache > self.cacheLimit:
            number_container = 0
            for app in self.applicationList:
                number_container += app.numberofContainers
            # fair cache allocation among containers, not applications
            cache_per_container = self.cacheLimit / number_container
            for app in self.applicationList:
                app.cacheThroughput = cache_per_container / app.cachePerQuery
        else:
            for app in self.applicationList:
                app.cacheThroughput = app.affinityThroughput

    def _update_tput(self):
        """
        calculate the throughput for each of the app residing in this node

        """
        nid = self.name
        nid_state = self.env.state[nid]
        cardinality_ratio = 1
        """
        Add additional affinity/anti-affinity
        """
        # if nid_state[7] + nid_state[8] >2:
        #     cardinality_ratio *= 1.2
        # if nid_state[0] + nid_state[1] >2:
        #     cardinality_ratio *= 1.2
        # if sum(nid_state) >15:
        #     cardinality_ratio *= 0.1
        # if sum(nid_state) <5:
        #     cardinality_ratio *= 1.2
        for app in self.applicationList:
            app.throughput = min(app.affinityThroughput, app.bwThroughput, app.memBwThroughput, app.cacheThroughput)
            app.throughput *= cardinality_ratio

    def total_tput(self):
        """
        return the summed throughput of all the containers residing in this node
        """
        tput = 0
        for app in self.applicationList:
            tput += app.throughput * app.numberofContainers
        return tput

    def minimum(self):
        # TODO: minimum throughput is not necessary based on 100
        minimum_tput = 100
        for app in self.applicationList:
            if app.throughput < minimum_tput:
                minimum_tput = app.throughput
        return minimum_tput

    def sla_violation(self,sla):
        # TODO: minimum throughput is not necessary based on 100
        violation = 0
        for app in self.applicationList:
            if app.throughput < sla:
                violation += app.numberofContainers
        return violation