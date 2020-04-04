import numpy as np
# import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Hyper-parameters
n_estimators=100
max_depth=20
random_state=0

def testing_with_other_dataset(origin='datasets/0507-all-110-results.npz', target='datasets/0508-low-105-results.npz', verbose=0):
    if verbose:
        print("Train with {}.\nTest  on   {}.".format(origin, target))
    npzfile = np.load(origin)
    x_ori, y1_ori, y2_ori, y3_ori = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    npzfile = np.load(target)
    x_tgt, y1_tgt, y2_tgt, y3_tgt = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    y3_ori = np.nan_to_num(y3_ori.astype(float))
    y3_tgt = np.nan_to_num(y3_tgt.astype(float))

    regr_ori = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
    cv_scores = cross_val_score(regr_ori, x_ori, y3_ori, cv=5, n_jobs=4)
    np.set_printoptions(precision=4, suppress=True)
    if verbose:
        print("5-fold cross validation scores of", origin, ":\n", cv_scores)

    regr_ori.fit(x_ori, y3_ori)
    score = regr_ori.score(x_tgt, y3_tgt)
    if verbose:
        print("R^2 score of regressor: %.4f" % score)

    return regr_ori


def training_with_cross_validation(npzfile_path='datasets/0507-all-110-results.npz', verbose=0):

    npzfile = np.load('./simulator/' + npzfile_path, allow_pickle=True)
    alloc, rt_50, rt_99, rps = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    length = len(rps)
    # for i in range(length):
    #     if alloc[i,0] * alloc[i,1] > 0:
    #         rps[i, 0] *= 0.5
    #     if alloc[i,5] * alloc[i,4] > 0:
    #         rps[i, 5] *= 2
    # for i in range(length):
    #     for j in range(6):
    #         # if rps[i, j] > 1:
    #         #     rps[i, j] *= 2.0
    #         if (rps[i, j] > 0) & (rps[i, j] < 1):
    #             rps[i, j] *= 0.5
    # for i in range(length):
    #     for j in range(6):
    #         if rps[i, j] > 1:
    #             if (j == 1) & (alloc[i, 0] * alloc[i, 1] > 0):
    #                 rps[i, j] *= 2.0
    #         if (rps[i, j] > 0) & (rps[i, j] < 1):
    #             rps[i, j] *= 0.5
    #: pre-processing
    rps = np.nan_to_num(rps.astype(float))
    X_train, X_test, y_train, y_test = train_test_split(alloc, rps, test_size=0.1, random_state=42)  # this random_state is not a hyper-parameter of Regressor
    if verbose:
        print("X_train {} => y_train {}".format(X_train.shape, y_train.shape))
        print("X_test  {} => y_test  {}".format(X_test.shape, y_test.shape))

    regr = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))

    cv_scores = cross_val_score(regr, X_train, y_train, cv=5, n_jobs=4)
    np.set_printoptions(precision=4, suppress=True)
    if verbose:
        print("5-fold cross validation scores:\n", cv_scores)

    regr.fit(X_train[:,0:7], y_train[:,0:7])
    score = regr.score(X_test[:,0:7], y_test[:,0:7])
    if verbose:
        print("R^2 score of regressor: %.4f" % score)
    return regr


def single_case_score(origin='datasets/0507-all-110-results.npz', target='datasets/0508-low-105-results.npz', threshold=0.8389):
    regr = testing_with_other_dataset(origin, target)

    npzfile = np.load(target)
    alloc, rt_50, rt_99, rps = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    rps = np.nan_to_num(rps.astype(float))

    np.set_printoptions(precision=4, suppress=True)
    for i in range(len(alloc)):
        score = regr.score([alloc[i]], [rps[i]])
        if abs(score - threshold) < 0.1:
            print("X     :",alloc[i])
            print("Y_pred:",regr.predict([alloc[i]])[0])
            print("Y_true:",rps[i])
            print("Score : %.4f\n" % score)


if __name__ == "__main__":
    npzfile_path_list = ['datasets/dataset-0423x2-0424x2-0425x2-median.npz', # 0
                         'datasets/0507-all-110-results.npz', # 1
                         'datasets/0508-low-105-results.npz', # 2
                         'datasets/0508-com-215-results.npz', # 3
                         'datasets/0509-low-110-results.npz', # 4
                         'datasets/0509-low-com-210-results.npz', # 5
                         'datasets/0509-com-325-results.npz', # 6
                         'datasets/0509-com-444-results.npz', # 7
                         'datasets/0601-mms-48-results.npz', #8
                         'datasets/0602-mss-revisit-135-results.npz', #9
                         'datasets/0627-8slot-503.npz', #10
                         'datasets/0701-rep-44.npz' #11
                         ]
    origin=npzfile_path_list[11]
    target=npzfile_path_list[10]


    # testing_with_other_dataset(origin=origin, target=target)
    # training_with_cross_validation(origin)
    single_case_score(origin, target, 0.6856)



""" Deprecated, use `testing_with_other_dataset`

def high_low_train_testing():
    npzfile = np.load('datasets/0507-all-110-results.npz')
    x_ori, y1_ori, y2_ori, y3_ori = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    npzfile = np.load('datasets/0508-low-105-results.npz')
    x_tgt, y1_tgt, y2_tgt, y3_tgt = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    y3_ori = np.nan_to_num(y3_ori.astype(float))
    y3_tgt = np.nan_to_num(y3_tgt.astype(float))

    #: ALL
    regr_ori = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
    ##: ALL CV
    cv_scores = cross_val_score(regr_ori, x_ori, y3_ori, cv=5, n_jobs=4)
    np.set_printoptions(precision=4, suppress=True)
    print("5-fold cross validation scores:\n", cv_scores)
    ##: ALL ON LOW
    regr_ori.fit(x_ori, y3_ori)
    score = regr_ori.score(x_tgt, y3_tgt)
    print("Train with ALL data. Test on LOW data.")
    print("R^2 score of regressor: %.4f" % score)

    #: LOW
    regr_tgt = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
    ##: LOW CV
    cv_scores = cross_val_score(regr_tgt, x_tgt, y3_tgt, cv=5, n_jobs=4)
    np.set_printoptions(precision=4, suppress=True)
    print("5-fold cross validation scores:\n", cv_scores)
    ##: LOW ON ALL
    regr_tgt.fit(x_tgt, y3_tgt)
    score = regr_tgt.score(x_ori, y3_ori)
    print("Train with LOW data. Test on ALL data.")
    print("R^2 score of regressor: %.4f" % score)
"""
