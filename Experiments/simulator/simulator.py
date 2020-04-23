# update: 2019-0902
# add Path, add in_xs.reshape(), pack as module

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from simulator.train_test import training_with_cross_validation
from simulator.utils import profile_series_to_profile_setting_list, profile_setting_list_to_reproduce_x

# # npzfile_path_list = ['datasets/0901-2046results-median-normed.npz'] # Latest dataset
# npzfile_path_list = ['datasets/0905-6435results-normed.npz']
# npzfile_path= Path(__file__).parent / npzfile_path_list[0]

def regularize_input_xs(input_xs): # 0714
    in_xs = None
    if type(input_xs) == np.ndarray:
        in_xs = input_xs.reshape(-1, 7) # reshape (9, ) to be (?, 9)
    elif type(input_xs) == str: # e.g., 8 slots -- "1-0-0-0-6-0-0-0 2-0-0-0-6-0-0-0 ..."
        profile_setting_list = profile_series_to_profile_setting_list(input_xs) # [[1,0,0,0,6,0,0,0], [2,0,0,0,6,0,0,0], ...]
        in_xs = profile_setting_list_to_reproduce_x(profile_setting_list)
    elif type(input_xs)==list: # e.g., 9 apps -- [[1, 0, 1, 0, 1, 1, 0, 1, 3]]
        if type(input_xs[0] != list): # e.g., 9 apps -- [1, 0, 1, 0, 1, 1, 0, 1, 3]
            in_xs = np.array([input_xs])
        else:
            in_xs = np.array(input_xs)
    else:
        print("Unsolved input of type: ", type(input_xs))
        print(input_xs)
    return in_xs


class Simulator:
    def __init__(self, npzfile_path='datasets/0905-2008results-median-normed.npz', verbose=0):
        # 100%: 0905-6435results-normed.
        # 20%:  0905-2008results-median-normed.npz
        # 5%:   1212-322results-normed.npz
        # 90%:  1212-5792results-normed.npz
        print("dataset: 20 percent")
        self.npzfile_path=npzfile_path
        # self.npzfile=np.load(npzfile_path)
        # self.alloc, self.rt_50, self.rt_99, self.rps=self.npzfile['alloc'], self.npzfile['rt_50'], self.npzfile['rt_99'], self.npzfile['rps']
        self.regr=training_with_cross_validation(npzfile_path, verbose=verbose)
        self.verbose=verbose

    def predict(self, input_xs):  # 0627
        in_xs = regularize_input_xs(input_xs) # 0714
        out_ys = None if in_xs is None else self.regr.predict(in_xs)
        try:
            return np.multiply(in_xs.astype(bool), out_ys)
            # element-wise multiply serving as quick fix for zero alloc.
        except:
            return out_ys

if __name__ == '__main__':
    #  microbench git:(master) $ python3 -m simulator.simulator
    # print("[INFO] Init with", npzfile_path, " ...")
    sim = Simulator(verbose=1)
    print("[INFO] Predict ...")
    input_xs = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1]
    ])
    print("[INFO]   Input:\n{}".format(input_xs))
    out_ys = sim.predict(input_xs)
    print("[INFO]   Output:\n{}".format(out_ys))