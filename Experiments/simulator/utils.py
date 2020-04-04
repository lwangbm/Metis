import numpy as np

def npzload(npzfile_path):
    npzfile = np.load(npzfile_path)
    alloc, rt_50, rt_99, rps=npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    rps = rps.astype('float')
    alloc_list = alloc.astype('int').tolist()
    return alloc_list, rps

def reproduce_x_to_profile_setting_list(array):  # 0627
    # input: array, shape: (?, 9), e.g., [[2. 1. 0. 4. 0. 1. 0. 0. 0.]]
    # output: [[1, 1, 2, 4, 4, 4, 4, 6]]
    summary_setting_list = None
    for item in array:
        setting = []
        for i in range(len(item)):
            if item[i] != 0:
                for num in range(int(item[i])):
                    setting.append(i+1)

        if len(setting) > 8: # over allocated
            print("[ERROR]: Too many containers: ", item, " => ", setting)
            break
        else:
            for i in range(8-len(setting)):
                setting.insert(0, 0) # add 0 in the first place

        # now the setting is well sorted
        if summary_setting_list is None:
          summary_setting_list = [setting]
        else:
          summary_setting_list.append(setting)
    return summary_setting_list

def profile_setting_list_to_reproduce_x(profile_setting_list):
    # input: list of list of 8 ints, e.g., [[1,0,0,0,6,0,0,0], [2,0,0,0,6,0,0,0], ...]
    # output: array, shape: (?, 9), e.g., [[1. 0. 0. 0. 0. 1. 0. 0. 0.], [0. 1. 0. 0. 0. 1. 0. 0. 0.]]
    array_list = []
    for profile_setting in profile_setting_list:
        item_array = np.zeros(9, dtype=int)
        for cntr in profile_setting:
            if cntr != 0:
                item_array[cntr-1] += 1 # index = cntr - 1
        array_list.append(item_array) # list.append is 10x faster than np.append
    return np.array(array_list)

def profile_setting_list_to_profile_series(summary_setting_list):
    return_string="PROFILE_SERIES=("
    for setting in summary_setting_list:
        setting_string='-'.join([str(i) for i in setting]) # '0-0-1-3-3-4-4-4'
        return_string += setting_string + ' '
    return_string = return_string[:-1] + ')' # remove last space
    return return_string

def profile_series_to_profile_setting_list(profile_series_string): # 8 slots -- "1-0-0-0-6-0-0-0 2-0-0-0-6-0-0-0 ..."
    if '(' in profile_series_string and ')' in profile_series_string: # "PROFILE_SERIES=(1-0-0-0-6-0-0-0 2-0-0-0-6-0-0-0 ...)"
        profile_series_string = profile_series_string.split('(')[1].split(')')[0] # "1-0-0-0-6-0-0-0 2-0-0-0-6-0-0-0 ..."
    profile_setting_list = []
    profile_series=profile_series_string.split()
    for profile_setting in profile_series:
        numbers = profile_setting.split('-')
        numbers_output = [int(x) for x in numbers]
        profile_setting_list.append(numbers_output) # [[1,0,0,0,6,0,0,0], [2,0,0,0,6,0,0,0], ...]
    return profile_setting_list


#############################
## print reproduce results ##
#############################

def query_results_multiple_scenarios(arr_in, npzfile_path='datasets/0706-nodup-1058.npz', num_scenarios=4, num_nodes=9, num_alloc=5):
    for scenario in range(num_scenarios):
        start_idx = scenario * num_nodes * num_alloc
        print('---- Scenario %d ----' % scenario)
        for alloc in range(num_alloc):
    #         print(start_idx, '->', start_idx + num_nodes)
            print('REPRODUCE:')
            query_results(arr_in[start_idx:start_idx + num_nodes], npzfile_path)
            start_idx += num_nodes
        print('=================\n')

def query_results_alloc_rps(query_allocation_array_list, alloc_list, rps, sim=None):
    print_alloc_string=''
    print_rps_string=''
    total_rps = 0
    for query_allocation_array in query_allocation_array_list:
        try:
            # query_allocation_array = reproduce_x_to_profile_setting([query_allocation_array])[0]
            if type(query_allocation_array) == list:
                index = alloc_list.index(query_allocation_array)
            else:
                index = alloc_list.index(query_allocation_array.tolist())
            rps_breakdown = rps[index]
            rps_sum = [ r * n for (r, n) in zip(rps_breakdown, query_allocation_array)]
            rps_sum = sum(np.nan_to_num(np.array(rps_sum)))
            total_rps += rps_sum
            # print("%s throughput: %.4f" % (query_allocation_array, rps_sum))
            print_alloc_string+=("%s throughput: %.4f\n" % (query_allocation_array, rps_sum))
            np.set_printoptions(precision=4, floatmode='fixed')
            # print("\t[rps/cntr]", np.nan_to_num(rps_breakdown))
            print_rps_string+=("[rps/cntr] %s\n" % np.nan_to_num(rps_breakdown))
        except ValueError:
            if sim is not None:
                rps_breakdown=sim.predict(np.array(query_allocation_array))
                rps_sum = [ r * n for (r, n) in zip(rps_breakdown, query_allocation_array)]
                total_rps += rps_sum
                print_alloc_string+=("%s throughput: %.4f\n" % (query_allocation_array, rps_sum))
                np.set_printoptions(precision=4, floatmode='fixed')
                # print("\t[rps/cntr]", np.nan_to_num(rps_breakdown))
                print_rps_string+=("[rps/cntr] %s\n" % np.nan_to_num(rps_breakdown))
            # print("%s throughput: nan" % (query_allocation_array))
            print_alloc_string+=("%s throughput: nan\n" % (query_allocation_array))
            not_found_setting=reproduce_x_to_profile_setting_list([query_allocation_array])[0]
            # print("\t[NOT FOUND] check: ", '-'.join([str(i) for i in not_found_setting]))
            print_rps_string+=("\t[NOT FOUND] check: %s\n" % ('-'.join([str(i) for i in not_found_setting])))
            rps_breakdown = [np.nan] * 9
            rps_sum = np.nan
    print("Total Throughput: %.4f" % total_rps)
    print(print_alloc_string)
    print(print_rps_string)


def query_results(query_allocation_array_list, npzfile_path='datasets/0706-nodup-1058.npz', sim=None):
    npzfile = np.load(npzfile_path)
    alloc, rt_50, rt_99, rps=npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    rps = rps.astype('float')
    alloc_list = alloc.tolist()

    query_results_alloc_rps(query_allocation_array_list, alloc_list, rps, sim)

def query_results_single_in_single_out(query_allocation_array, npzfile_path='datasets/0706-nodup-1058.npz'):
    npzfile = np.load(npzfile_path)
    alloc, rt_50, rt_99, rps=npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    rps = rps.astype('float')
    alloc_list = alloc.tolist()
    query_allocation_array = query_allocation_array.flatten()
    try:
        if type(query_allocation_array) == list:
            index = alloc_list.index(query_allocation_array)
        else:
            index = alloc_list.index(query_allocation_array.tolist())
        rps_breakdown = rps[index]
        return np.nan_to_num(rps_breakdown).reshape(1, -1)
    except ValueError: # not found
        return None
############

def ndarray_match(x1, x2, as_int=True, verbose=1):
    """
    input:
        ndarray_match(x1, x2, True, 1)
    print:                      return:
        x1[4]==x2[4]                { 4: 4,
        x1[11]==x2[99]               11: 99}
    """
    if as_int:
        x1=x1.astype(int)
        x2=x2.astype(int)
    list_x1=x1.tolist()
    list_x2=x2.tolist()
    # only list has index implemented.
    # numpy.tolist() preserves the order

    match_dict={}
    for x1_index in range(len(list_x1)):
        item = list_x1[x1_index]
        try:
            x2_index = list_x2.index(item)
            if verbose:
                print("x1[{}]==x2[{}]".format(x1_index, x2_index))
            match_dict[x1_index] = x2_index
        except ValueError:
            pass
    return match_dict


def display_matched_items(x1, x2, y1, y2, top=None):
    match_dict = ndarray_match(x1, x2)
    try:
        match_dict_items = list(match_dict.items())[:top]
    except:
        match_dict_items = match_dict.items()
    for (i,j) in match_dict_items:
        print(x1[i],x2[j])
        print(y1[i])
        print(y2[j])
        print()


def find_self_duplicate(x, verbose=1):
    """
    print:                                  return:
    12 [0. 1. 0. 1. 0. 0. 0. 1. 5.] 4           {12: 4,
    15 [1. 1. 1. 1. 0. 0. 0. 0. 4.] 14           15: 14}
    """
    temp_set=set()
    duplicate_dict={}
    try:
        x = x.tolist()
    except:
        pass
    for index in range(len(x)):
        item_list = x[index]
        item_str = str(item_list)
        if item_str in temp_set:
            if verbose:
                first_occur_index = x.index(item_list)
                print(index, x[index], first_occur_index)
            duplicate_dict[index] = first_occur_index
        temp_set.add(item_str)
    return duplicate_dict

def remove_self_duplicate(x, y, verbose=1):
    output_x=[]
    output_y=[]
    try:
        x = x.tolist()
        y = y.tolist()
    except:
        pass
    for index in range(len(x)):
        item_x = x[index]
        item_y = y[index]
        if item_x not in output_x:
            output_x.append(item_x)
            output_y.append(item_y)
        elif verbose:
            print(index, item_x, output_x.index(item_x))
    return output_x, output_y

def remove_self_duplicate_compare_rps(x, y, y1, y2, y_concern_item=None, verbose=1):
    output_x=[]
    output_y=[]
    output_y1=[]
    output_y2=[]
    try:
        x = x.tolist()
        y = y.tolist()
        y1 = y1.tolist()
        y2 = y2.tolist()
    except:
        pass
    for index in range(len(x)):
        item_x = x[index]
        item_y = y[index]
        item_y1 = y1[index]
        item_y2 = y2[index]
        if item_x in output_x:
            if y_concern_item is not None: # compare, e.g., concern_item=5 => FileIO rps
                master_id = output_x.index(item_x)
                master_y = output_y[master_id]
                if item_y[y_concern_item] >= master_y[y_concern_item]:
                    output_y[master_id] = y[index].copy()
                    output_y1[master_id] = y1[index].copy()
                    output_y2[master_id] = y2[index].copy()
                    if verbose:
                        print('Replace:', index, item_x, '\n   ', master_y, '\n-->', output_y[master_id])
                else:
                    pass # challenge fails
                    if verbose:
                        print('Discard:', index, item_x, output_x.index(item_x))
            else:
                if verbose:
                    print('Discard:', index, item_x, output_x.index(item_x))
        else:
            output_x.append(item_x)
            output_y.append(item_y)
            output_y1.append(item_y1)
            output_y2.append(item_y2)
    return np.array(output_x), np.array(output_y), np.array(output_y1), np.array(output_y2)

def add_zero_allocation_to_dataset(x, y, y1, y2):
    output_x = np.vstack([x, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
    output_y = np.vstack([y, np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])])
    output_y1 = np.vstack([y1, np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])])
    output_y2 = np.vstack([y2, np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])])
    return output_x, output_y, output_y1, output_y2


def combine_two_npzfiles(one=None, two=None, com=None):
  if one is None or two is None or com is None:
    print("one + two => com. Should not be None.")

  npzfile = np.load(one)
  x_one, y1_one, y2_one, y3_one = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']

  npzfile = np.load(two)
  x_two, y1_two, y2_two, y3_two = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']

  x_com = np.append(x_one, x_two, axis=0)
  y1_com = np.append(y1_one, y1_two, axis=0)
  y2_com = np.append(y2_one, y2_two, axis=0)
  y3_com = np.append(y3_one, y3_two, axis=0)

  print("ONE: ", len(x_one),len(y1_one),len(y2_one),len(y3_one))
  print("TWO: ", len(x_two),len(y1_two),len(y2_two),len(y3_two))
  print("COM: ", len(x_com),len(y1_com),len(y2_com),len(y3_com))
  print("Save to: ", com)

  np.savez(com, alloc=x_com, rt_50=y1_com, rt_99=y2_com, rps=y3_com)


def median_of_duplicated_records(a34, r34, rt5034, rt9934):
    for j in range(len(a34)):
        indices=[i for i, x in enumerate(a34) if x == a34[j]]
        if len(indices) > 1:
            rps_single = np.median(r34[indices], axis=0)
            rt50_single = np.median(rt5034[indices], axis=0)
            rt99_single = np.median(rt9934[indices], axis=0)
            for k in indices:
                r34[k] = rps_single
                rt5034[k] = rt50_single
                rt9934[k] = rt99_single
    return a34, r34, rt5034, rt9934

##########################
## deprecated functions ##
##########################

def r1r2m1m2_to_mms_local_remote(input_xs):
    if len(input_xs.shape) == 1:
        output_x = r1r2m1m2_to_mms_local_remote_1d(input_xs)
        return output_x.reshape(-1, 9)
    if len(input_xs.shape) > 2:
        input_xs = input_xs.reshape(-1, 9)  # going throughput the following

    output_xs = []
    for input_x in input_xs:
        output_x = r1r2m1m2_to_mms_local_remote_1d(input_x)
        output_xs.append(output_x)
    output_xs = np.array(output_xs)
    return output_xs

def r1r2m1m2_to_mms_local_remote_1d(input_x):
    """
    input_x  = [1, 0, 1, 0, 1, 1, 0, 1, 3] # R1, R2, M1, M2, XXXXX
    output_x = [0, 1, 1, 0, 1, 1, 0, 1, 3] # CPU, Redis, MMS_LOCAL, MMS_REMOTE, XXXXX
    """
    output_x = input_x.copy()  # input_x[-5:] unchanged

    num_redis = input_x[0] + input_x[1]  # Redis = R1 + R2
    num_mms_local = 0
    num_mms_remote = 0

    if input_x[0] >= 1:  # if R1 exists
        num_mms_local += input_x[2]  # MMS_LOCAL += R1
    else:
        num_mms_remote += input_x[2]  # MMS_REMOTE += R1

    if input_x[1] >= 1:  # if R2 exists
        num_mms_local += input_x[3]  # MMS_LOCAL += R2
    else:
        num_mms_remote += input_x[3]  # MMS_REMOTE += R2

    output_x[0] = 0  # Sysbench CPU records discarded
    output_x[1] = num_redis
    output_x[2] = num_mms_local
    output_x[3] = num_mms_remote

    return output_x

def mms_local_remote_to_r1r2m1m2(input_xs, input_ys):
    if len(input_xs.shape) != len(input_ys.shape):
        print("Shape of Input X and Y does not match")
        exit()
    if len(input_xs.shape) == 1:
        output_y = mms_local_remote_to_r1r2m1m2_1d(input_xs, input_ys)
        return output_y.reshape(-1, 9)
    if len(input_xs.shape) > 2:
        input_xs = input_xs.reshape(-1, 9)  # going throughput the following
        input_ys = input_ys.reshape(-1, 9)  # going throughput the following

    output_ys = []
    for index in range(input_xs.shape[0]):
        input_x = input_xs[index]
        input_y = input_ys[index]
        output_y = mms_local_remote_to_r1r2m1m2_1d(input_x, input_y)
        output_ys.append(output_y)
    output_ys = np.array(output_ys)
    return output_ys

def mms_local_remote_to_r1r2m1m2_1d(input_x, input_y):
    """
    input_x  = [1, 0, 1, 0, 1, 1, 0, 1, 3] # R1, R2, M1, M2, XXXXX
    output_x = [0, 1, 1, 0, 1, 1, 0, 1, 3] # CPU, Redis, MMS_LOCAL, MMS_REMOTE, XXXXX

    input_y  = [0.0000, 0.8218, 4.0633, 0.0000, 2.6703, 1.0034, 0.0000, 0.8029, 0.5026] # CPU, Redis, MMS_LOCAL, MMS_REMOTE, XXXXX
    output_y = [0.8218, 0.0000, 4.0633, 0.0000, 2.6703, 1.0034, 0.0000, 0.8029, 0.5026] # R1, R2, M1, M2, XXXXX
    """
    output_y = input_y.copy()  # input_y[-5:] unchanged

    if input_x[0] >= 1:  # if R1 exists
        output_y[0] = input_y[1]  # R1 = Redis
        if input_x[2] >= 1:  # if R1 & M1 co-exists
            output_y[2] = input_y[2]  # M1 = MMS_LOCAL
        else:
            output_y[2] = 0  # M1 = 0
    else:  # if R1 not exists
        output_y[0] = 0  # R1 = 0
        if input_x[2] >= 1:  # if M1 exists
            output_y[2] = input_y[3]  # M1 = MMS_REMOTE
        else:
            output_y[2] = 0  # M1 = 0

    if input_x[1] >= 1:  # if R2 exists
        output_y[1] = input_y[1]  # R2 = Redis
        if input_x[3] >= 1:  # if R2 & M2 co-exists
            output_y[3] = input_y[2]  # M1 = MMS_LOCAL
        else:
            output_y[3] = 0  # M2 = 0
    else:  # if R2 not exists
        output_y[1] = 0  # R1 = 0
        if input_x[3] >= 1:  # if M2 exists
            output_y[3] = input_y[3]  # M2 = MMS_REMOTE
        else:
            output_y[3] = 0  # M1 = 0

    return output_y




#
