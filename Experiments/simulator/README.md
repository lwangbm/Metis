# README

## Usage (06.12)
**TL;DR**: in python under `microbench/simulator`
```python
from simulator import Simulator
sim = Simulator()
out_ys = sim.predict( [[1, 0, 1, 0, 1, 1, 0, 1, 3]] )
print(out_ys)
# [[0.7665 0.     4.     0.     2.5498 0.3787 0.     0.7208 0.4363]]

# input format:
# x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7] , x[0][8]
# Redis-1, Redis-2, MMS-1,   MMS-2,  Memory(Sysbench), FileIO(Sysbench), CPU(Stress-ng), Memory(Stress-ng), FileIO(Stress-ng)
```

With more details
```python
import numpy as np
from simulator import Simulator
npzfile_path='datasets/0612-mms_combine-298.npz'  # see datasets/ for more .npz
npzfile = np.load(npzfile_path)
x, y1, y2, y3 = npzfile['alloc'], npzfile['rt_50'], npzfile['rt_99'], npzfile['rps']

sim = Simulator(npzfile_path, verbose=1)
out_ys, ys = sim._predict_with_ys(x[:10])

print("\nPrediction vs. Ground Truth:")
print(ys)      # CPU, Redis, MMS_LOCAL, MMS_REMOTE, ...
print(y3[:10]) # CPU, Redis, MMS_LOCAL, MMS_REMOTE, ...

print("\nThe final RPS output:")
print(out_ys)  # Redis-1, Redis-2, MMS-1, MMS-2, ...
```

## Dataset scores

- 0610-comine-438.npz
    - X_train (394, 9) => y_train (394, 9); X_test  (44, 9) => y_test  (44, 9)
    - 5-fold cross validation scores: [0.8385 0.8923 0.812  0.7998 0.8711]
    - R^2 score of regressor: 0.8854
- 0611-mms-175.npz
    - X_train (157, 9) => y_train (157, 9); X_test  (18, 9) => y_test  (18, 9)
    - 5-fold cross validation scores: [0.7514 0.7164 0.8142 0.7609 0.6741]
    - R^2 score of regressor: -2.3604
- 0612-438_no_mms-141.npz
    - X_train (126, 9) => y_train (126, 9); X_test  (15, 9) => y_test  (15, 9)
    - 5-fold cross validation scores: [0.6842 0.8509 0.6235 0.7773 0.7749]
    - R^2 score of regressor: 0.8975
- **0612-mms_combine-298.npz** (recommended)
    - X_train (268, 9) => y_train (268, 9); X_test  (30, 9) => y_test  (30, 9)
    - 5-fold cross validation scores: [0.8353 0.8525 0.8805 0.9009 0.8153]
    - R^2 score of regressor: 0.8628

## Workload description

| Index | Workload           | RPS  |
| ----- | ------------------ | ---- |
| 0     | Redis-1            |      |
| 1     | Redis-2            |      |
| 2     | MMS-1 (=> Redis-1) |      |
| 3     | MMS-2 (=> Redis-2) |      |
| 4     | Memory (Sysbench)  |      |
| 5     | FileIO (Sysbench)  |      |
| 6     | CPU (Stress-ng)    |      |
| 7     | Memory (Stress-ng) |      |
| 8     | FileIO (Stress-ng) |      |

## Update remarks (06.12)

1. 把 simulator 封装成了 class，可以无脑调用 sim = Simulator(); sim.predict( ...... )

2. 把 dataset 重新整理了一下，目前只用关心 4 个就好，其他都放在 datasets/deprecated/ 下了
    1. 0610-combine-438.npz: 清理了之前 444 dataset 中一些错误的条目；
    2. 0611-mms-175.npz: 包含最新的 MXNet Model Serving 的数据集；
    3. 0612-438_no_mms-141.npz: 共141条，删除了原来 444 中的旧版的 MXNet Model Serving (MMS) workload
    4. 0612-mms_combine-298.npz: 合并了 2.2 和 2.3，并修改了有冲突的条目 （推荐使用这个数据集）

3. 根据上次的讨论，修改了 workload，在原本的 9 个 application 中：
    1. 第 1 项 CPU 被删除
    2. 前 4 项被改为 Redis-1, Redis-2, MMS-1, MMS-2。MMS-1 只有和 Redis-1 放在一起时才会加速，和 Redis-2 放在一起时无加速效果；MMS-2 与 Redis-2 的关系同理。
    3. 后 5 项原封不动
    4. 原 workload 和新 workload 的映射关系（包括部署的映射，和结果 RPS 的反映射），已经写在了 Simulator 类中，可以直接调用 predict 函数。



## Usage (06.03)

- Open `dataset-sklearn.py`
- Revise L#118 and L#119 to set different dataset, according to L#107
- L#122: Basic training using the original dataset
- L#121: Transfer learning: testing the accuracy of the regressor, which is trained from original dataset, on the target dataset
- L#123: deprecated function
- L#124: showing study cases of transfer learning, displaying how good/bad is the prediction, compared with ground truth.



## General Workflow

1. load xxx.npz as  `data`  (as L#18-19 in `dataset-sklearn.py`)
2. Build regressor `regr ` (as L#25)
3. regressor.fit(`data.x`, `data.y`) (as L#30)
   1. y1: 50% Response time
   2. y2: 99% Response time
   3. y3: Requests per second
4. calculate score (as #L32)

---

- Updated results

```bash
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
```



---


- Trained Model applied on Different Datasets
```
Train with 0509-com-444-results.npz.
Test  on   dataset-0423x2-0424x2-0425x2-median.npz.
5-fold cross validation scores of 0509-com-444-results.npz :
 [0.5826 0.5426 0.6628 0.6041 0.6707]
R^2 score of regressor: 0.8820

Train with 0509-com-444-results.npz.
Test  on   0507-all-110-results.npz.
5-fold cross validation scores of 0509-com-444-results.npz :
 [0.5826 0.5426 0.6628 0.6041 0.6707]
R^2 score of regressor: 0.9263

Train with 0509-com-444-results.npz.
Test  on   0509-low-com-210-results.npz.
5-fold cross validation scores of 0509-com-444-results.npz :
 [0.5826 0.5426 0.6628 0.6041 0.6707]
R^2 score of regressor: 0.9129
```

---

- 0507-all-110-results.npz:
```
X_train (99, 9) => y_train (99, 9)
X_test  (11, 9) => y_test  (11, 9)
5-fold cross validation scores:
 [0.5325 0.6933 0.6966 0.5624 0.1777]
R^2 score of regressor:  0.7532
```


- 0508-low-105-results.npz
```
X_train (94, 9) => y_train (94, 9)
X_test  (11, 9) => y_test  (11, 9)
5-fold cross validation scores:
 [0.7067 0.8153 0.8695 0.8763 0.6878]
R^2 score of regressor:  0.8459
```

- 0508-com-215-results.npz:
```
X_train (193, 9) => y_train (193, 9)
X_test  (22, 9) => y_test  (22, 9)
5-fold cross validation scores:
 [0.8394 0.795  0.7246 0.821  0.7647]
R^2 score of regressor: 0.7633
```

- Cross dataset testing:
```
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
```
