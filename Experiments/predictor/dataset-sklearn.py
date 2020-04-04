import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Hyper-parameters
n_estimators=20
max_depth=10
random_state=0

def testing_with_other_dataset(origin='0507-all-110-results.npz', target='0508-low-105-results.npz'):
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
    print("5-fold cross validation scores of", origin, ":\n", cv_scores)

    regr_ori.fit(x_ori, y3_ori)
    score = regr_ori.score(x_tgt, y3_tgt)
    print("R^2 score of regressor: %.4f" % score)

    return regr_ori

def high_low_train_testing():
    npzfile = np.load('0507-all-110-results.npz')
    x_ori, y1_ori, y2_ori, y3_ori = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']
    npzfile = np.load('0508-low-105-results.npz')
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

def training_with_cross_validation(npzfile_path='0507-all-110-results.npz'):

    npzfile = np.load(npzfile_path)
    alloc, rt_50, rt_99, rps = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']

    #: pre-processing
    rps = np.nan_to_num(rps.astype(float))
    X_train, X_test, y_train, y_test = train_test_split(alloc, rps, test_size=0.1, random_state=42)  # this random_state is not a hyper-parameter of Regressor
    print("X_train {} => y_train {}".format(X_train.shape, y_train.shape))
    print("X_test  {} => y_test  {}".format(X_test.shape, y_test.shape))

    regr = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))

    cv_scores = cross_val_score(regr, X_train, y_train, cv=5, n_jobs=4)
    np.set_printoptions(precision=4, suppress=True)
    print("5-fold cross validation scores:\n", cv_scores)

    regr.fit(X_train, y_train)
    score = regr.score(X_test, y_test)
    print("R^2 score of regressor: %.4f" % score)


def single_case_score(origin='0507-all-110-results.npz', target='0508-low-105-results.npz', threshold=0.8389):
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


npzfile_path_list = ['dataset-0423x2-0424x2-0425x2-median.npz', # 0
                     '0507-all-110-results.npz', # 1
                     '0508-low-105-results.npz', # 2
                     '0508-com-215-results.npz', # 3
                     '0509-low-110-results.npz', # 4
                     '0509-low-com-210-results.npz', # 5
                     '0509-com-325-results.npz', # 6
                     '0509-com-444-results.npz', # 7
                     ]
origin=npzfile_path_list[2]
target=npzfile_path_list[4]

# testing_with_other_dataset(origin=origin, target=target)
# training_with_cross_validation(origin)
# high_low_train_testing()
single_case_score(origin, target, 0.6856)


# Each one's single training
""" 0507-all-110-results.npz:
X_train (99, 9) => y_train (99, 9)
X_test  (11, 9) => y_test  (11, 9)
5-fold cross validation scores:
 [0.5325 0.6933 0.6966 0.5624 0.1777]
R^2 score of regressor:  0.7532
"""

""" 0508-low-105-results.npz:
X_train (94, 9) => y_train (94, 9)
X_test  (11, 9) => y_test  (11, 9)
5-fold cross validation scores:
 [0.7067 0.8153 0.8695 0.8763 0.6878]
R^2 score of regressor:  0.8459
"""

""" 0509-low-110-results.npz:
X_train (99, 9) => y_train (99, 9)
X_test  (11, 9) => y_test  (11, 9)
5-fold cross validation scores:
 [ 0.8316  0.8654 -2.2003  0.9044  0.8574]
R^2 score of regressor: 0.8377
"""

""" 0509-low-com-210-results.npz:
X_train (193, 9) => y_train (193, 9)
X_test  (22, 9) => y_test  (22, 9)
5-fold cross validation scores:
 [0.8498 0.7929 0.8373 0.758  0.808 ]
R^2 score of regressor: 0.8585
"""

""" 0508-com-215-results.npz:
X_train (193, 9) => y_train (193, 9)
X_test  (22, 9) => y_test  (22, 9)
5-fold cross validation scores:
 [0.8394 0.795  0.7246 0.821  0.7647]
R^2 score of regressor: 0.7633
"""

""" 0509-com-325-results.npz:
X_train (292, 9) => y_train (292, 9)
X_test  (33, 9) => y_test  (33, 9)
5-fold cross validation scores:
 [0.8421 0.8315 0.7095 0.815  0.8361]
R^2 score of regressor: 0.7602
"""

""" dataset-0423x2-0424x2-0425x2-median.npz
X_train (107, 9) => y_train (107, 9)
X_test  (12, 9) => y_test  (12, 9)
5-fold cross validation scores:
 [0.8178 0.7822 0.7528 0.7574 0.6664]
R^2 score of regressor: 0.7690
"""

""" 0509-com-444-results.npz
X_train (399, 9) => y_train (399, 9)
X_test  (45, 9) => y_test  (45, 9)
5-fold cross validation scores:
 [0.804  0.8303 0.8931 0.8083 0.8586]
R^2 score of regressor: 0.8906
"""


# Testing on others
""" 
0507-all-110-results.npz:
5-fold cross validation scores:
 [ 0.7878  0.7126 -3.2436  0.7069  0.4705]
Train with ALL data. Test on LOW data.
R^2 score of regressor: -2.0701

0508-low-105-results.npz:
5-fold cross validation scores:
 [0.8579 0.8736 0.1086 0.7433 0.8783]
Train with LOW data. Test on ALL data.
R^2 score of regressor: -0.1895
"""

"""
Train with 0507-all-110-results.npz.
Test  on   dataset-0423x2-0424x2-0425x2-median.npz.
5-fold cross validation scores of 0507-all-110-results.npz :
 [ 0.7878  0.7126 -3.2436  0.7069  0.4705]
R^2 score of regressor: 0.3914
"""

"""
Train with 0508-com-215-results.npz.
Test  on   dataset-0423x2-0424x2-0425x2-median.npz.
5-fold cross validation scores of 0508-com-215-results.npz :
 [ 0.5875  0.633   0.7865 -0.2698 -0.0077]
R^2 score of regressor: 0.1309
"""

"""
Train with 0509-com-325-results.npz.
Test  on   dataset-0423x2-0424x2-0425x2-median.npz.
5-fold cross validation scores of 0509-com-325-results.npz :
 [0.5388 0.7702 0.1803 0.6861 0.455 ]
R^2 score of regressor: 0.1051
"""