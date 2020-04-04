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
