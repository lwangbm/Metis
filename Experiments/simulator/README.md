# README

**Usage**: in python under `/Metis/Experiments`
```python
import numpy as np
from simulator.simulator import Simulator
npzfile_path='datasets/0905-2008results-median-normed.npz'  # see datasets/ for more .npz

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

sim = Simulator(npzfile_path, verbose=1)
out_ys = sim.predict(input_xs)

print("\nThe final RPS output:")
print(out_ys)  # Redis, MMS, Checksum, ISR, YCSB-A, YCSB-B, SceneDetect
```