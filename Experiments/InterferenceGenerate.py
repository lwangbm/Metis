import numpy as np
import sys
import pandas as pd
sys.path.append("/Users/ourokutaira/Desktop/Metis")
from cluster_env import LraClusterEnv


params = {
        'predictor_path': "trace_set_1",
        'predicted_metric': 'cpi',
        'container_limitation per node': 8,
        'alloc_path': './simulator/datasets/0905-7app-2008-pairwise-normed.csv',
        'rps_path': './simulator/datasets/0905-7app-2008-pairwise-normed-per-app.csv'

    }


def train(params):

    import pandas as pd
    df = pd.read_csv(params['alloc_path'], sep=',', header=0)
    allocation = df.values[:,0:7]
    df = pd.read_csv(params['rps_path'], sep=',', header=0)
    rps = df.values
    env = LraClusterEnv(num_nodes=9)
    capacity = params['container_limitation per node']
    NUM_APP = 7
    miss_sample = 0
    state_set = np.empty((0, 4), int)
    rps_set = np.empty((0, capacity), float)
    num_cardinality = 0
    for s_tag in range(NUM_APP):
        container_list = np.zeros([1, NUM_APP])
        container_list[0, s_tag] += 1
        exist = (allocation == container_list[0]).all(1).any()
        if exist:
            tput_breakdown_single = rps[allocation.tolist().index(container_list[0].tolist())]
        else:
            tput_node, tput_breakdown_single = (env.get_throughput_given_state(container_list))
            tput_breakdown_single = tput_breakdown_single[0]
            miss_sample += 1
        tput_s_tag_original = tput_breakdown_single[s_tag]
        for c_tag in range (NUM_APP):
            tput_s_tag_set = []
            for num_c_tag in range(0,capacity):
                container_list = np.zeros([1, NUM_APP])
                container_list[0, s_tag] += 1
                container_list[0, c_tag] += num_c_tag
                exist = (allocation == container_list[0]).all(1).any()
                if exist:
                    tput_breakdown_single = rps[allocation.tolist().index(container_list[0].tolist())]

                else:
                    tput_node, tput_breakdown_single = (env.get_throughput_given_state(container_list))
                    tput_breakdown_single = tput_breakdown_single[0]
                    miss_sample += 1
                tput_s_tag = tput_breakdown_single[s_tag]
                tput_s_tag_set.append(tput_s_tag)
            if np.max(tput_s_tag_set) / np.min(tput_s_tag_set) > 1.1:
                for cehck_num in range(1, capacity):
                    if tput_s_tag_set[cehck_num]/tput_s_tag_original> 1.1: #1.4
                        interfence_tag = 1  # larger than
                        state_set = np.append(state_set, np.array([s_tag+1, c_tag+1, interfence_tag, cehck_num]).reshape([1, 4]), axis=0)
                        rps_set = np.append(rps_set, np.array(tput_s_tag_set / tput_s_tag_original).reshape([1, 8]),
                                            axis=0)
                        break
                    if tput_s_tag_set[cehck_num]/tput_s_tag_original< 0.9: #0.6:
                        interfence_tag = 0  # less than
                        if c_tag == s_tag:
                            cehck_num+=1
                        state_set = np.append(state_set, np.array([s_tag+1, c_tag+1, interfence_tag, cehck_num-1]).reshape([1, 4]), axis=0)
                        rps_set = np.append(rps_set, np.array(tput_s_tag_set / tput_s_tag_original).reshape([1, 8]),
                                            axis=0)
                        break
            rise = 0
            fall = 0
            for cehck_num in range(1, capacity):
                if tput_s_tag_set[cehck_num] / tput_s_tag_set[cehck_num-1] > 1.2:
                    rise = 1
                if tput_s_tag_set[cehck_num] / tput_s_tag_set[cehck_num-1] < 0.8:
                    fall = 1
            if rise*fall > 0:
                num_cardinality += 1
                print(s_tag+1, c_tag+1)

    import pandas as pd
    save = pd.DataFrame(state_set,columns=["app_s","app_c", "less_or_lager", "threshold"])
    save.to_csv('./interference_applist.csv', index=False, header=True)

    save_1 = pd.DataFrame(rps_set,columns=[0,1,2,3,4,5,6,7])
    save_1.to_csv('./interference_rpslist.csv', index=False, header=True)
    print("num_cardinality: %d"%num_cardinality)
    print("miss_sample:", miss_sample)


def main():

    train(params)


if __name__ == "__main__":
    main()
